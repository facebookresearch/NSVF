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

from fairnr.data.data_utils import Timer, GPUTimer
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.fairnr_model import BaseModel

MAX_DEPTH = 10000.0


@register_model('nsvf')
class NSVFModel(BaseModel):

    READER = 'image_reader'
    ENCODER = 'sparsevoxel_encoder'
    FIELD = 'radiance_field'
    RAYMARCHER = 'volume_rendering'

    def _forward(self, ray_start, ray_dir, **kwargs):
        S, V, P, _ = ray_dir.size()
        assert S == 1, "naive NeRF only supports single object."

        # voxel encoder (precompute for each voxel if needed)
        encoder_states = self.encoder.precompute(**kwargs)  

        # ray-voxel intersection
        with GPUTimer() as timer0:
            ray_start, ray_dir, intersection_outputs, hits = \
                self.encoder.ray_intersect(ray_start, ray_dir, encoder_states)

            if self.reader.no_sampling and self.training:  # sample points after ray-voxel intersection
                uv, size = kwargs['uv'], kwargs['size']
                mask = hits.reshape(*uv.size()[:2], uv.size(-1))

                # sample rays based on voxel intersections
                sampled_uv, sampled_masks = self.reader.sample_pixels(
                    uv, size, mask=mask, return_mask=True)
                sampled_masks = sampled_masks.reshape(uv.size(0), -1).bool()
                hits, sampled_masks = hits[sampled_masks].reshape(S, -1), sampled_masks.unsqueeze(-1)
                intersection_outputs = {name: outs[sampled_masks.expand_as(outs)].reshape(S, -1, outs.size(-1)) 
                    for name, outs in intersection_outputs.items()}
                ray_start = ray_start[sampled_masks.expand_as(ray_start)].reshape(S, -1, 3)
                ray_dir = ray_dir[sampled_masks.expand_as(ray_dir)].reshape(S, -1, 3)
                P = hits.size(-1) // V   # the number of pixels per image
            else:
                sampled_uv = None
        
        # neural ray-marching
        fullsize = S * V * P
        
        BG_DEPTH = self.field.bg_color.depth
        bg_color = self.field.bg_color(ray_dir)
        
        all_results = defaultdict(lambda: None)
        if hits.sum() > 0:  # check if ray missed everything
            intersection_outputs = {name: outs[hits] for name, outs in intersection_outputs.items()}
            ray_start, ray_dir = ray_start[hits], ray_dir[hits]
            
            # sample evalution points along the ray
            samples = self.encoder.ray_sample(intersection_outputs)
            encoder_states = {name: s.reshape(-1, s.size(-1)) if s is not None else None
                for name, s in encoder_states.items()}
            
            # rendering
            all_results = self.raymarcher(
                self.encoder, self.field, ray_start, ray_dir, samples, encoder_states)
            all_results['depths'] = all_results['depths'] + BG_DEPTH * all_results['missed']
            all_results['voxel_edges'] = self.encoder.get_edge(ray_start, ray_dir, samples, encoder_states)
            all_results['voxel_depth'] = samples['sampled_point_depth'][:, 0]

        # fill out the full size
        hits = hits.reshape(fullsize)
        all_results['missed'] = fill_in((fullsize, ), hits, all_results['missed'], 1.0).view(S, V, P)
        all_results['depths'] = fill_in((fullsize, ), hits, all_results['depths'], BG_DEPTH).view(S, V, P)
        all_results['voxel_depth'] = fill_in((fullsize, ), hits, all_results['voxel_depth'], BG_DEPTH).view(S, V, P)
        all_results['voxel_edges'] = fill_in((fullsize, 3), hits, all_results['voxel_edges'], 1.0).view(S, V, P, 3)
        all_results['colors'] = fill_in((fullsize, 3), hits, all_results['colors'], 0.0).view(S, V, P, 3)
        all_results['bg_color'] = bg_color.reshape(fullsize, 3).view(S, V, P, 3)
        all_results['colors'] += all_results['missed'].unsqueeze(-1) * all_results['bg_color']
        if 'normal' in all_results:
            all_results['normal'] = fill_in((fullsize, 3), hits, all_results['normal'], 0.0).view(S, V, P, 3)

        # other logs
        all_results['other_logs'] = {
                'voxs_log': self.encoder.voxel_size.item(),
                'stps_log': self.encoder.step_size.item(),
                'tvox_log': timer0.sum,
                'asf_log': (all_results['ae'].float() / fullsize).item(),
                'ash_log': (all_results['ae'].float() / hits.sum()).item(),
                'nvox_log': self.encoder.num_voxels,
                }
        all_results['sampled_uv'] = sampled_uv
        return all_results

    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        images = super()._visualize(images, sample, output, state, **kwargs)
        if 'voxel_edges' in output and output['voxel_edges'] is not None:
            # voxel hitting visualization
            images['{}_voxel/{}:HWC'.format(name, img_id)] = {
                'img': output['voxel_edges'][shape, view].float(), 
                'min_val': 0, 
                'max_val': 1,
                'weight':
                    compute_normal_map(
                        output['ray_start'][shape, view].float(),
                        output['ray_dir'][shape, view].float(),
                        output['voxel_depth'][shape, view].float(),
                        sample['extrinsics'][shape, view].float().inverse(),
                        width, proj=True)
                }
        if 'normal' in output and output['normal'] is not None:
            images['{}_predn/{}:HWC'.format(name, img_id)] = {
                'img': output['normal'][shape, view], 'min_val': -1, 'max_val': 1}
        return images
    
    @torch.no_grad()
    def prune_voxels(self, th=0.5, train_stats=False):
        self.encoder.pruning(self.field, th, train_stats=train_stats)
        self.clean_caches()

    @torch.no_grad()
    def split_voxels(self):
        logger.info("half the global voxel size {:.4f} -> {:.4f}".format(
            self.encoder.voxel_size.item(), self.encoder.voxel_size.item() * .5))
        self.encoder.splitting()
        self.encoder.voxel_size *= .5
        self.encoder.max_hits *= 1.5
        self.clean_caches()

    @torch.no_grad()
    def reduce_stepsize(self):
        logger.info("reduce the raymarching step size {:.4f} -> {:.4f}".format(
            self.encoder.step_size.item(), self.encoder.step_size.item() * .5))
        self.encoder.step_size *= .5

    def clean_caches(self, reset=False):
        self.encoder.clean_runtime_caches()
        if reset:
            self.encoder.reset_runtime_caches()

@register_model_architecture("nsvf", "nsvf_base")
def base_architecture(args):
    # parameter needs to be changed
    args.voxel_size = getattr(args, "voxel_size", 0.25)
    args.max_hits = getattr(args, "max_hits", 60)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
    args.raymarching_stepsize_ratio = getattr(args, "raymarching_stepsize_ratio", 0.0)
    
    # encoder default parameter
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 32)
    args.voxel_path = getattr(args, "voxel_path", None)
    args.initial_boundingbox = getattr(args, "initial_boundingbox", None)

    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32")
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
    args.use_octree = getattr(args, "use_octree", False)

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


@register_model_architecture("nsvf", "nsvf_xyz")
def nerf2_architecture(args):
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 0)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, ray:4")
    base_architecture(args)


@register_model_architecture("nsvf", "nsvf_xyzn")
def nerf3_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, normal:4, ray:4")
    nerf2_architecture(args)


@register_model_architecture("nsvf", "nsvf_embn")
def nerf4_architecture(args):
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, normal:4, ray:4")
    base_architecture(args)


@register_model_architecture("nsvf", "nsvf_emb0")
def nerf5_architecture(args):
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 384)
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:0:384")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    base_architecture(args)


@register_model('rnsvf')
class ResampledNSVFModel(NSVFModel):

    RAYMARCHER = "resampled_volume_rendering"


@register_model_architecture("rnsvf", "rnsvf_base")
def rnsvf_architecture(args):
    base_architecture(args)


@register_model('disco_nsvf')
class ResampledNSVFModel(NSVFModel):

    FIELD = "disentangled_radiance_field"


@register_model_architecture("disco_nsvf", "disco_nsvf")
def disco_nsvf_architecture(args):
    args.compressed_light_dim = getattr(args, "compressed_light_dim", 64)
    nerf3_architecture(args)
