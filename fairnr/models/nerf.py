# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)

import fairnr.clib as _C
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.fairnr_model import BaseModel


@register_model('nerf')
class NeRFModel(BaseModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    READER = 'image_reader'
    FIELD = 'radiance_field'
    RAYMARCHER = 'volume_rendering'

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--near', type=float, help='near distance of the volume')
        parser.add_argument('--far',  type=float, help='far distance of the volume')
        parser.add_argument('--fixed-num-samples', type=int, 
            help='number of samples for the first pass along the ray.')
        parser.add_argument('--fixed-fine-num-samples', type=int,
            help='sample a fixed number of points for each ray in hierarchical sampling, e.g. 64, 128.')
        parser.add_argument('--reduce-fine-for-missed', action='store_true',
            help='if set, the number of fine samples is discounted based on foreground probability only.')

    def preprocessing(self, **kwargs):
        return {}  # we do not use encoder for NeRF

    def intersecting(self, ray_start, ray_dir, encoder_states, **kwargs):
        S, V, P, _ = ray_dir.size()
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        intersection_outputs = {
            "min_depth": ray_dir.new_ones(S, V * P, 1) * self.args.near,
            "max_depth": ray_dir.new_ones(S, V * P, 1) * self.args.far,
            "probs": ray_dir.new_ones(S, V * P, 1),
            "steps": ray_dir.new_ones(S, V * P, 1) * self.args.fixed_num_samples,
            "intersected_voxel_idx": ray_dir.new_zeros(S, V * P, 1).int()}
        hits = ray_dir.new_ones(S, V * P).bool()
        return ray_start, ray_dir, intersection_outputs, hits, None

    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        # sample points and use middle point approximation
        sampled_idx, sampled_depth, sampled_dists = _C.inverse_cdf_sampling(
            intersection_outputs['intersected_voxel_idx'], 
            intersection_outputs['min_depth'], 
            intersection_outputs['max_depth'], 
            intersection_outputs['probs'],
            intersection_outputs['steps'], -1, (not self.training))
        from fairseq import pdb;pdb.set_trace()
        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,  # dummy index (to match raymarcher)
        }
        field = self.field_fine if fine and (self.field_fine is not None) else self.field 
        field_input_fn = lambda samples, encoder_states: {
            'pos': samples['sampled_point_xyz'].requires_grad_(True),
            'ray': samples['sampled_point_ray_direction']}
        all_results = self.raymarcher(
            field_input_fn, field, ray_start, ray_dir, samples, encoder_states
        )
        return samples, all_results

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        # this function is basically the same as that in NSVF model.
        intersection_outputs['min_depth'] = samples['sampled_point_depth'] - samples['sampled_point_distance'] * .5
        intersection_outputs['max_depth'] = samples['sampled_point_depth'] + samples['sampled_point_distance'] * .5
        intersection_outputs['intersected_voxel_idx'] = samples['sampled_point_voxel_idx'].contiguous()

        safe_probs = all_results['probs'] + 1e-8  # HACK: make a non-zero distribution
        intersection_outputs['probs'] = safe_probs / safe_probs.sum(-1, keepdim=True)
        intersection_outputs['steps'] = intersection_outputs['steps'] * 0 + self.args.fixed_fine_num_samples
        if getattr(self.args, "reduce_fine_for_missed", False):
            intersection_outputs['steps'] = intersection_outputs['steps'] * safe_probs.sum(-1)
        from fairseq import pdb;pdb.set_trace()
        return intersection_outputs

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # vanilla nerf hits everything. so no need to fill_in
        S, V, P = sizes
        fullsize = S * V * P
        
        all_results['missed'] = all_results['missed'].view(S, V, P)
        all_results['colors'] = all_results['colors'].view(S, V, P, 3)
        all_results['depths'] = all_results['depths'].view(S, V, P)

        BG_DEPTH = self.field.bg_color.depth
        bg_color = self.field.bg_color(all_results['colors'])
        all_results['colors'] += all_results['missed'].unsqueeze(-1) * bg_color.reshape(fullsize, 3).view(S, V, P, 3)
        all_results['depths'] += all_results['missed'] * BG_DEPTH
        if 'normal' in all_results:
            all_results['normal'] = all_results['normal'].view(S, V, P, 3)
        return all_results

    def add_other_logs(self, all_results):
        return {}


@register_model_architecture("nerf", "nerf_base")
def base_architecture(args):
    # parameter needs to be changed
    args.near = getattr(args, "near", 2)
    args.far = getattr(args, "far", 4)
    args.fixed_num_samples = getattr(args, "fixed_num_samples", 64)
    args.fixed_fine_num_samples = getattr(args, "fixed_fine_num_samples", 64)
    args.hierarchical_sampling = getattr(args, "hierarchical_sampling", True)
    args.use_fine_model = getattr(args, "use_fine_model", True)

    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.background_depth = getattr(args, "background_depth", 5.0)
    
    # raymarcher
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)
    args.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0)

    # reader
    args.pixel_per_view = getattr(args, "pixel_per_view", 2048)
    args.sampling_on_mask = getattr(args, "sampling_on_mask", 0.0)
    args.sampling_at_center = getattr(args, "sampling_at_center", 1.0)
    args.sampling_on_bbox = getattr(args, "sampling_on_bbox", False)
    args.sampling_patch_size = getattr(args, "sampling_patch_size", 1)
    args.sampling_skipping_size = getattr(args, "sampling_skipping_size", 1)

    # others
    args.chunk_size = getattr(args, "chunk_size", 64)
    args.valid_chunk_size = getattr(args, "valid_chunk_size", 64)
