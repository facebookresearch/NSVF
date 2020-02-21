# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import math

import torch.nn.functional as F
import torch
from torch import Tensor

from fairseq import metrics
from fairseq.utils import item
from fairseq.criterions import FairseqCriterion, register_criterion

import fairdr.criterions.utils as utils


class RenderingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        pass

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample)
        loss, loss_output = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = 1
        
        logging_output = {
            'loss': loss.data.item() if reduce else loss.data,
            'nsentences': sample['alpha'].size(0),
            'ntokens':  sample['alpha'].size(1),
            'npixels': sample['alpha'].size(2),
            'sample_size': sample_size,
        }
        for w in loss_output:
            logging_output[w] = loss_output[w]
        
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        raise NotImplementedError

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        
        summed_logging_outputs = {
            w: sum(log.get(w, 0) for log in logging_outputs)
            for w in logging_outputs[0]
        }
        sample_size = summed_logging_outputs['sample_size']
        
        for w in summed_logging_outputs:
            if '_loss' in w:
                metrics.log_scalar(w[:3], summed_logging_outputs[w] / sample_size, sample_size, round=3)
            if w == 'loss':
                metrics.log_scalar('loss', summed_logging_outputs['loss'] / sample_size, sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


@register_criterion('srn_loss')
class SRNLossCriterion(RenderingCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        if args.vgg_weight > 0:
            from fairdr.criterions.perceptual_loss import VGGPerceptualLoss
            self.vgg = VGGPerceptualLoss(resize=True)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--L1', action='store_true',
                            help='if enabled, use L1 instead of L2 for RGB loss')
        parser.add_argument('--rgb-weight', type=float, default=200.0)
        parser.add_argument('--reg-weight', type=float, default=1e-3)
        parser.add_argument('--depth-weight', type=float, default=0.0)
        parser.add_argument('--gp-weight', type=float, default=0.0)
        parser.add_argument('--vgg-weight', type=float, default=0.0)
        parser.add_argument('--error-map', action='store_true')

    def compute_loss(self, model, net_output, sample, reduce=True):

        alpha  = (sample['alpha'] > 0.5)
        masks  = torch.ones_like(alpha)
        
        losses = {}
        rgb_loss = utils.rgb_loss(
            net_output['predicts'], sample['rgb'], 
            masks, self.args.L1)
        losses['rgb_loss'] = (rgb_loss, self.args.rgb_weight)

        if self.args.reg_weight > 0:
            reg_loss = utils.depth_regularization_loss(
                net_output['depths'])
            losses['reg_loss'] = (reg_loss, self.args.reg_weight)

        if self.args.depth_weight > 0 and sample['depths'] is not None:
            depth_loss = utils.depth_loss(net_output['depths'], sample['depths'], masks)
            losses['depth_loss'] = (depth_loss, self.args.depth_weight)

        if self.args.gp_weight > 0:
            losses['grd_loss'] = (net_output['grad_penalty'], self.args.gp_weight)

        if self.args.vgg_weight > 0:
            if sample.get('full_rgb', None) is None:
                target = sample['rgb'] 
                inputs = net_output['predicts']
            else:
                target = sample['full_rgb']
                inputs = target.scatter(
                    2, sample['index'].unsqueeze(-1).expand_as(net_output['predicts']),
                    net_output['predicts'])

            def transform(x):
                S, V, D, _ = x.size()
                L = int(math.sqrt(D))
                x = x.transpose(2, 3).view(S * V, 3, L, L)
                return x / 2 + 0.5

            losses['vgg_loss'] = (self.vgg(transform(inputs), transform(target)), self.args.vgg_weight)
        
        loss = sum(losses[key][0] * losses[key][1] for key in losses)
        return loss, {key: item(losses[key][0]) for key in losses}