# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairnr.modules.linear import FCLayer
from fairnr.data.geometry import ray

MAX_DEPTH = 10000.0


class Renderer(nn.Module):
    """
    Abstract class for ray marching
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError


class VolumeRenderer(Renderer):

    def __init__(self, args):
        super().__init__(args) 
        self.chunk_size = 256 * getattr(args, "chunk_size", 256)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0.0)

    @staticmethod
    def add_args(parser):
        # ray-marching parameters
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')
        
        # additional arguments
        parser.add_argument('--chunk-size', type=int, metavar='D', 
                            help='set chunks to go through the network. trade time for memory')
        parser.add_argument('--raymarching-tolerance', type=float, default=0)
    
    def forward_once(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states, early_stop=None):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        sampled_depth, sampled_idx, sampled_dists = samples
        sampled_idx = sampled_idx.long()
        
        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if early_stop is not None:
            sample_mask = sample_mask & (~early_stop.unsqueeze(-1))
        
        if sample_mask.sum() == 0:  # miss everything skip
            return torch.zeros_like(sampled_depth), sampled_depth.new_zeros(*sampled_depth.size(), 3)

        queries = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        queries = queries[sample_mask]
        querie_dirs = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])[sample_mask]
        sampled_idx = sampled_idx[sample_mask]
        sampled_dists = sampled_dists[sample_mask]

        # get encoder features as inputs
        field_inputs = input_fn((queries, sampled_idx), encoder_states)

        # forward implicit fields
        sigma, texture = field_fn(field_inputs, dir=querie_dirs)
        
        # post processing
        noise = 0 if not self.discrete_reg and (not self.training)  else torch.zeros_like(sigma).normal_()  
        free_energy = torch.relu(noise + sigma) * sampled_dists    # (optional) free_energy = (F.elu(sigma - 3, alpha=1) + 1) * dists
        free_energy = torch.zeros_like(sampled_depth).masked_scatter(sample_mask, free_energy)
        texture = free_energy.new_zeros(*free_energy.size(), 3).masked_scatter(
            sample_mask.unsqueeze(-1).expand(*sample_mask.size(), 3), texture)
        return free_energy, texture

    def forward(self, input_fn, field_fn, ray_start, ray_dir, samples, encoder_states, gt_depths=None):
        sampled_depth, sampled_idx, sampled_dists = samples
        tolerance = self.raymarching_tolerance
        early_stop = None
        if tolerance > 0:
            tolerance = -math.log(tolerance)
           
        hits = sampled_idx.ne(-1).long()
        free_energy, texture = [], []
        size_so_far, start_step = 0, 0
        accumulated_free_energy = 0
        
        for i in range(hits.size(1) + 1):
            if ((i == hits.size(1)) or (size_so_far + hits[:, i].sum() > self.chunk_size)) and (i > start_step):
                _free_energy, _texture = self.forward_once(
                        input_fn, field_fn, 
                        ray_start, ray_dir, (
                        sampled_depth[:, start_step: i], 
                        sampled_idx[:, start_step: i], 
                        sampled_dists[:, start_step: i]), 
                        encoder_states, 
                        early_stop=early_stop)

                accumulated_free_energy += _free_energy.sum(1)
                if tolerance > 0:
                    early_stop = accumulated_free_energy > tolerance
                    hits[early_stop] *= 0

                free_energy += [_free_energy]
                texture += [_texture]
                start_step, size_so_far = i, 0
            
            if (i < hits.size(1)):
                size_so_far += hits[:, i].sum()
        
        free_energy = torch.cat(free_energy, 1)
        texture = torch.cat(texture, 1)

        # aggregate along the ray
        shifted_free_energy = torch.cat([free_energy.new_zeros(sampled_depth.size(0), 1), free_energy[:, :-1]], dim=-1)  # shift one step
        a = 1 - torch.exp(-free_energy.float())                             # probability of it is not empty here
        b = torch.exp(-torch.cumsum(shifted_free_energy.float(), dim=-1))   # probability of everything is empty up to now
        probs = (a * b).type_as(free_energy)                                # probability of the ray hits something here
        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)
        rgb = (texture * probs.unsqueeze(-1)).sum(-2)
        
        # additional loss on the variance of depth prediction
        var_loss = ((sampled_depth ** 2 * probs).sum(-1) - depth ** 2).mean()
        return rgb, depth, missed, var_loss

