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
        super().__init__(task)
        self.args = args

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

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
            elif '_weight' in w:
                metrics.log_scalar('w_' + w[:3], summed_logging_outputs[w] / sample_size, sample_size, round=3)
            elif '_acc' in w:
                metrics.log_scalar('a_' + w[:3], summed_logging_outputs[w] / sample_size, sample_size, round=3)
            elif w == 'loss':
                metrics.log_scalar('loss', summed_logging_outputs['loss'] / sample_size, sample_size, priority=0, round=3)
            elif '_log' in w:
                metrics.log_scalar(w[:3], summed_logging_outputs[w] / sample_size, sample_size, priority=1, round=3)

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
            self._dummy = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), 
                        requires_grad=True)  # HACK: to avoid warnings in c10d

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--L1', action='store_true',
                            help='if enabled, use L1 instead of L2 for RGB loss')
        parser.add_argument('--rgb-weight', type=float, default=200.0)
        parser.add_argument('--reg-weight', type=float, default=1e-3)
        parser.add_argument('--max-depth', type=float, default=5)
        parser.add_argument('--depth-weight', type=float, default=0.0)
        parser.add_argument('--depth-weight-decay', type=str, default=None,
                            help="""if set, use tuple to set (final_ratio, steps).
                                    For instance, (0, 30000)    
                                """)
        parser.add_argument('--freespace-weight', type=float, default=0.0)
        parser.add_argument('--occupancy-weight', type=float, default=0.0)
        parser.add_argument('--entropy-weight', type=float, default=0.0)
        parser.add_argument('--pruning-weight', type=float, default=0.0)
        parser.add_argument('--alpha-weight', type=float, default=0.0)
        parser.add_argument('--gp-weight', type=float, default=0.0)
        parser.add_argument('--vgg-weight', type=float, default=0.0)
        parser.add_argument('--vgg-level', type=int, choices=[1,2,3,4], default=2)
        parser.add_argument('--error-map', action='store_true')
        parser.add_argument('--no-loss-if-predicted', action='store_true',
                            help="if set, loss is only on the predicted pixels.")
        parser.add_argument('--max-margin-freespace', type=float, default=None,
                            help="instead of binary classification. use margin-loss")
        parser.add_argument('--no-background-loss', action='store_true',
                            help="do not compute RGB-loss on the background.")
        parser.add_argument('--random-background-loss', action='store_true',
                            help='set if we are using transparent image')

    def compute_loss(self, model, net_output, sample, reduce=True):
        alpha  = (sample['alpha'] > 0.5)
        masks = alpha.clone() if self.args.no_background_loss else torch.ones_like(alpha)
        
        losses, other_logs = {}, {}
        if 'other_logs' in net_output:
            other_logs.update(net_output['other_logs'])
        if not self.args.random_background_loss:
            rgb_loss = utils.rgb_loss(
                net_output['predicts'], sample['rgb'], 
                masks, self.args.L1)
        else:
            random_bg = torch.zeros_like(net_output['predicts']).uniform_(-1, 1) * 0.1 + net_output['bg_color'].unsqueeze(0) * 0.9
            predicts = net_output['predicts'] + \
                net_output['missed'].unsqueeze(-1) * (random_bg - net_output['bg_color'].unsqueeze(0))
            targets = sample['rgb'].masked_scatter(~(alpha.unsqueeze(-1).expand_as(random_bg)), random_bg)
            rgb_loss = utils.rgb_loss(predicts, targets, 
                masks, self.args.L1)
        
        losses['rgb_loss'] = (rgb_loss, self.args.rgb_weight)

        if self.args.reg_weight > 0:
            if 'latent' in net_output:
                losses['reg_loss'] = (net_output['latent'], self.args.reg_weight)

            else:
                min_depths = net_output.get('min_depths', 0.0)
                reg_loss = utils.depth_regularization_loss(
                    net_output['depths'], min_depths)
                losses['reg_loss'] = (reg_loss, 10000.0 * self.args.reg_weight)

        if self.args.entropy_weight > 0:
            losses['ent_loss'] = (net_output['entropy'], self.args.entropy_weight)

        if self.args.alpha_weight > 0:
            alpha = net_output['missed'].reshape(-1)
            # alpha_loss = torch.log(0.1 + alpha) + torch.log(0.1 + 1 - alpha) - math.log(0.11)
            # alpha_loss = alpha_loss.float().mean().type_as(alpha_loss)
            alpha_loss = torch.log1p(
                1. / 0.11 * alpha.float() * (1 - alpha.float())
            ).mean().type_as(alpha)
            losses['alpha_loss'] = (alpha_loss, self.args.alpha_weight)

        if self.args.depth_weight > 0:
            if sample['depths'] is not None:
                depth_mask = masks & (sample['depths'] > 0)
                depth_loss = utils.depth_loss(net_output['depths'], sample['depths'], depth_mask)
                
            else:
                # no depth map is provided. depth loss only applied on background.
                max_depth_target = self.args.max_depth * torch.ones_like(net_output['depths'])
                if sample['mask'] is not None:        
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, (1 - sample['mask']).bool())
                else:
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, ~alpha)
            
            depth_weight = self.args.depth_weight
            if self.args.depth_weight_decay is not None:
                final_factor, final_steps = eval(self.args.depth_weight_decay)
                depth_weight *= max(0, 1 - (1 - final_factor) * self.task._num_updates / final_steps)
                other_logs['depth_weight'] = depth_weight

            losses['depth_loss'] = (depth_loss, depth_weight)

        if (self.args.freespace_weight > 0) and (self.args.occupancy_weight > 0):
            freespace_mask = (1 - sample['mask']).bool() if sample['mask'] is not None else ~alpha
            occupancy_mask = sample['mask'].bool() if sample['mask'] is not None else alpha
            # if self.args.no_loss_if_predicted:
            #     freespace_mask = freespace_mask & (net_output['missed'] < 0)
            if self.args.max_margin_freespace is None:
                freespace_pred = 1 - torch.sigmoid(net_output['missed'] / 0.1)
                freespace_loss = utils.space_loss(freespace_pred, freespace_mask, sum=False)
                
                occupancy_pred = 1 - torch.sigmoid(net_output['missed'] / 0.1)
                if self.args.no_loss_if_predicted:
                    occupancy_pred = occupancy_pred.masked_fill(net_output['missed'] < 0, 1.0)
                occupancy_loss = utils.occupancy_loss(occupancy_pred, occupancy_mask, sum=False)
            
            else:
                margin = self.args.max_margin_freespace
                assert self.args.no_loss_if_predicted, "only support no loss if preidcted"

                freespace_loss, occupancy_loss = utils.maxmargin_space_loss(
                    net_output['missed'], freespace_mask, margin, sum=False
                )

            losses['freespace_loss'] = (freespace_loss, self.args.freespace_weight)
            losses['occupancy_loss'] = (occupancy_loss, self.args.occupancy_weight)    
     
            other_logs['freespace_acc'] = ((net_output['missed'] > 0)[freespace_mask]).float().mean()
            other_logs['occupancy_acc'] = ((net_output['missed'] < 0)[occupancy_mask]).float().mean()

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
                H, W = int(sample['size'][0, 0, 0]), int(sample['size'][0, 0, 1])
                x = x.transpose(2, 3).view(S * V, 3, H, W)
                return x / 2 + 0.5

            # vgg_ratio = target.numel() / net_output['predicts'].numel()
            losses['vgg_loss'] = (self.vgg(
                transform(inputs), transform(target), self.args.vgg_level) + 0.0 * self._dummy, self.args.vgg_weight)

        if self.args.pruning_weight > 0:
            assert 'pruning_loss' in net_output, "requires pruning loss to be computed."
            losses['pruning_loss'] = (net_output['pruning_loss'], self.args.pruning_weight)

        loss = sum(losses[key][0] * losses[key][1] for key in losses)
        logging_outputs = {key: item(losses[key][0]) for key in losses}
        logging_outputs.update(other_logs)
        return loss, logging_outputs