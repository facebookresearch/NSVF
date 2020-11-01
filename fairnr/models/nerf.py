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

from fairnr.models.fairnr_model import BaseModel


@register_model('nerf')
class NeRFModel(BaseModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    ENCODER = 'volume_encoder'
    READER = 'image_reader'
    FIELD = 'radiance_field'
    RAYMARCHER = 'volume_rendering'

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--fixed-num-samples', type=int, 
            help='number of samples for the first pass along the ray.')
        parser.add_argument('--fixed-fine-num-samples', type=int,
            help='sample a fixed number of points for each ray in hierarchical sampling, e.g. 64, 128.')
        parser.add_argument('--reduce-fine-for-missed', action='store_true',
            help='if set, the number of fine samples is discounted based on foreground probability only.')

    def preprocessing(self, **kwargs):
        return self.encoder.precompute(**kwargs)

    def intersecting(self, ray_start, ray_dir, encoder_states, **kwargs):
        ray_start, ray_dir, intersection_outputs, hits = \
            self.encoder.ray_intersect(ray_start, ray_dir, encoder_states)
        return ray_start, ray_dir, intersection_outputs, hits, None

    def raymarching(self, ray_start, ray_dir, intersection_outputs, encoder_states, fine=False):
        # sample points and use middle point approximation
        samples = self.encoder.ray_sample(intersection_outputs)
        field = self.field_fine if fine and (self.field_fine is not None) else self.field 
        all_results = self.raymarcher(
            self.encoder, field, ray_start, ray_dir, samples, encoder_states
        )
        return samples, all_results

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        # this function is basically the same as that in NSVF model.
        depth = samples.get('original_point_depth', samples['sampled_point_depth'])
        dists = samples.get('original_point_distance', samples['sampled_point_distance'])
        intersection_outputs['min_depth'] = depth - dists * .5
        intersection_outputs['max_depth'] = depth + dists * .5
        intersection_outputs['intersected_voxel_idx'] = samples['sampled_point_voxel_idx'].contiguous()
        safe_probs = all_results['probs'] + 1e-8  # HACK: make a non-zero distribution
        intersection_outputs['probs'] = safe_probs / safe_probs.sum(-1, keepdim=True)
        intersection_outputs['steps'] = safe_probs.new_ones(*safe_probs.size()[:-1], 1) 
        if getattr(self.args, "fixed_fine_num_samples", 0) > 0:
            intersection_outputs['steps'] = intersection_outputs['steps'] * self.args.fixed_fine_num_samples
        if getattr(self.args, "reduce_fine_for_missed", False):
            intersection_outputs['steps'] = intersection_outputs['steps'] * safe_probs.sum(-1, keepdim=True)
        return intersection_outputs

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # vanilla nerf hits everything. so no need to fill_in
        S, V, P = sizes
        fullsize = S * V * P
        
        all_results['missed'] = all_results['missed'].view(S, V, P)
        all_results['colors'] = all_results['colors'].view(S, V, P, 3)
        all_results['depths'] = all_results['depths'].view(S, V, P)
        if 'z' in all_results:
            all_results['z'] = all_results['z'].view(S, V, P)

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


@register_model('sg_nerf')
class SGNeRFModel(NeRFModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    ENCODER = 'infinite_volume_encoder'

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # vanilla nerf hits everything. so no need to fill_in
        S, V, P = sizes
        all_results['missed'] = all_results['missed'].view(S, V, P)
        all_results['colors'] = all_results['colors'].view(S, V, P, 3)
        all_results['depths'] = all_results['depths'].view(S, V, P)
        if 'z' in all_results:
            all_results['z'] = all_results['z'].view(S, V, P)
        if 'normal' in all_results:
            all_results['normal'] = all_results['normal'].view(S, V, P, 3)
        return all_results

@register_model_architecture("sg_nerf", "sg_nerf_base")
def sg_nerf_architecture(args):
    INF_FAR = 1e6
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10:4")
    args.near = getattr(args, "near", 2)
    args.far = getattr(args, "far", INF_FAR)
    base_architecture(args)


@register_model('sg2_nerf')
class SG2NeRFModel(SGNeRFModel):
    """ This is a simple re-implementation of the vanilla NeRF
    """
    ENCODER = 'stereographic_volume_encoder'

    def prepare_hierarchical_sampling(self, intersection_outputs, samples, all_results):
        # this function is basically the same as that in NSVF model.
        depth = samples['original_point_depth']
        dists = samples['original_point_distance']
        intersection_outputs['min_theta'] = depth - dists * .5
        intersection_outputs['max_theta'] = depth + dists * .5
        intersection_outputs['intersected_voxel_idx'] = samples['sampled_point_voxel_idx'].contiguous()
        safe_probs = all_results['probs'] + 1e-8  # HACK: make a non-zero distribution
        intersection_outputs['probs'] = safe_probs / safe_probs.sum(-1, keepdim=True)
        intersection_outputs['steps'] = safe_probs.new_ones(*safe_probs.size()[:-1], 1) 
        if getattr(self.args, "fixed_fine_num_samples", 0) > 0:
            intersection_outputs['steps'] = intersection_outputs['steps'] * self.args.fixed_fine_num_samples
        if getattr(self.args, "reduce_fine_for_missed", False):
            intersection_outputs['steps'] = intersection_outputs['steps'] * safe_probs.sum(-1, keepdim=True)
        return intersection_outputs

@register_model_architecture("sg2_nerf", "sg2_nerf_base")
def sg2_nerf_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10:4")
    args.near = getattr(args, "near", 2)
    base_architecture(args)
