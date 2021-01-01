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
import fairnr.criterions.utils as utils

class RenderingCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(task)
        self.args = args
        self.hierarchical = getattr(args, 'hierarchical_loss', False)

    @classmethod
    def build_criterion(cls, args, task):
        """Construct a criterion from command-line args."""
        return cls(args, task)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--hierarchical-loss', action='store_true',
            help='if set, it computes both the coarse and fine-level losses in hierarchical sampling.')


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample)
        sample.update(net_output['samples'])
    
        loss, loss_output = self.compute_loss(model, net_output, sample, reduce=reduce)
        if self.hierarchical:
            assert net_output.get('coarse', None) is not None, "missing coarse level outputs."
            loss0, loss_output0 = self.compute_loss(model, net_output['coarse'], sample, reduce=reduce)
            loss = loss + loss0
            loss_output.update({'cor-' + key: loss_output0[key] for key in loss_output0})
        
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
                metrics.log_scalar(w.split('_')[0], summed_logging_outputs[w] / sample_size, sample_size, round=3)
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
        # HACK: to avoid warnings in c10d
        self.dummy_loss = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=True)  
        if args.vgg_weight > 0:
            from fairnr.criterions.perceptual_loss import VGGPerceptualLoss
            self.vgg = VGGPerceptualLoss(resize=False)

        if args.eval_lpips:
            from lpips_pytorch import LPIPS
            self.lpips = LPIPS(net_type='alex', version='0.1')
            
    @staticmethod
    def add_args(parser):
        RenderingCriterion.add_args(parser)
        parser.add_argument('--L1', action='store_true',
                            help='if enabled, use L1 instead of L2 for RGB loss')
        parser.add_argument('--color-weight', type=float, default=256.0)
        parser.add_argument('--depth-weight', type=float, default=0.0)
        parser.add_argument('--depth-weight-decay', type=str, default=None,
                            help="""if set, use tuple to set (final_ratio, steps).
                                    For instance, (0, 30000)    
                                """)
        parser.add_argument('--alpha-weight', type=float, default=0.0)
        parser.add_argument('--vgg-weight', type=float, default=0.0)
        parser.add_argument('--eikonal-weight', type=float, default=0.0)
        parser.add_argument('--regz-weight', type=float, default=0.0)
        parser.add_argument('--vgg-level', type=int, choices=[1,2,3,4], default=2)
        parser.add_argument('--eval-lpips', action='store_true',
                            help="evaluate LPIPS scores in validation")
        parser.add_argument('--no-background-loss', action='store_true')
        
    def compute_loss(self, model, net_output, sample, reduce=True):
        losses, other_logs = {}, {}
    
        # prepare data before computing loss
        sampled_uv = sample['sampled_uv']  # S, V, 2, N, P, P (patch-size)
        S, V, _, N, P1, P2 = sampled_uv.size()
        H, W, h, w = sample['size'][0, 0].long().cpu().tolist()
        L = N * P1 * P2
        flatten_uv = sampled_uv.view(S, V, 2, L)
        flatten_index = (flatten_uv[:,:,0] // h + flatten_uv[:,:,1] // w * W).long()

        assert 'colors' in sample and sample['colors'] is not None, "ground-truth colors not provided"
        target_colors = sample['colors']
        masks = (sample['alpha'] > 0) if self.args.no_background_loss else None
        if L < target_colors.size(2):    
            target_colors = target_colors.gather(2, flatten_index.unsqueeze(-1).repeat(1,1,1,3))
            masks = masks.gather(2, flatten_uv) if masks is not None else None
    
        if 'other_logs' in net_output:
            other_logs.update(net_output['other_logs'])

        # computing loss
        if self.args.color_weight > 0:
            color_loss = utils.rgb_loss(
                net_output['colors'], target_colors, 
                masks, self.args.L1)
            losses['color_loss'] = (color_loss, self.args.color_weight)
        
        if self.args.alpha_weight > 0:
            _alpha = net_output['missed'].reshape(-1)
            alpha_loss = torch.log1p(
                1. / 0.11 * _alpha.float() * (1 - _alpha.float())
            ).mean().type_as(_alpha)
            losses['alpha_loss'] = (alpha_loss, self.args.alpha_weight)

        if self.args.depth_weight > 0:
            if sample['depths'] is not None:
                target_depths = target_depths.gather(2, flatten_index)
                depth_mask = masks & (target_depths > 0)
                depth_loss = utils.depth_loss(net_output['depths'], target_depths, depth_mask)
                
            else:
                # no depth map is provided, depth loss only applied on background based on masks
                max_depth_target = self.args.max_depth * torch.ones_like(net_output['depths'])
                if sample['mask'] is not None:        
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, (1 - sample['mask']).bool())
                else:
                    depth_loss = utils.depth_loss(net_output['depths'], max_depth_target, ~masks)
            
            depth_weight = self.args.depth_weight
            if self.args.depth_weight_decay is not None:
                final_factor, final_steps = eval(self.args.depth_weight_decay)
                depth_weight *= max(0, 1 - (1 - final_factor) * self.task._num_updates / final_steps)
                other_logs['depth_weight'] = depth_weight

            losses['depth_loss'] = (depth_loss, depth_weight)

        
        if self.args.vgg_weight > 0:
            assert P1 * P2 > 1, "we have to use a patch-based sampling for VGG loss"
            target_colors = target_colors.reshape(-1, P1, P2, 3).permute(0, 3, 1, 2) * .5 + .5
            output_colors = net_output['colors'].reshape(-1, P1, P2, 3).permute(0, 3, 1, 2) * .5 + .5
            vgg_loss = self.vgg(output_colors, target_colors)
            losses['vgg_loss'] = (vgg_loss, self.args.vgg_weight)

        if self.args.eikonal_weight > 0:
            losses['eik_loss'] = (net_output['eikonal-term'].mean(), self.args.eikonal_weight)
        
        # if self.args.regz_weight > 0:
        losses['reg_loss'] = (net_output['regz-term'].mean(), self.args.regz_weight)
        loss = sum(losses[key][0] * losses[key][1] for key in losses)
       
        # add a dummy loss
        loss = loss + model.dummy_loss + self.dummy_loss * 0.
        logging_outputs = {key: item(losses[key][0]) for key in losses}
        logging_outputs.update(other_logs)
        return loss, logging_outputs
