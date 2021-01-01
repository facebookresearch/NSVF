# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger(__name__)

import cv2, math, time, copy, json
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairseq.utils import item, with_torch_seed
from fairnr.data.geometry import compute_normal_map, fill_in
from fairnr.models.nsvf import NSVFModel, base_architecture, nerf_style_architecture
from fairnr.models.fairnr_model import get_encoder, get_field, get_reader, get_renderer

@register_model('nsvf_bg')
class NSVFBGModel(NSVFModel):

    def __init__(self, args, setups):
        super().__init__(args, setups)

        args_copy = copy.deepcopy(args)
        if getattr(args, "bg_field_args", None) is not None:
            args_copy.__dict__.update(json.loads(args.bg_field_args))
        else:
            args_copy.inputs_to_density = "pos:10"
            args_copy.inputs_to_texture = "feat:0:256, ray:4:3:b"
        self.bg_field  = get_field("radiance_field")(args_copy)
        self.bg_encoder = get_encoder("volume_encoder")(args_copy)

    @classmethod
    def add_args(cls, parser):
        super().add_args(parser)
        parser.add_argument('--near', type=float, help='near distance of the volume')
        parser.add_argument('--far',  type=float, help='far distance of the volume')
        parser.add_argument('--nerf-steps', type=int, help='additional nerf steps')
        parser.add_argument('--bg-field-args', type=str, default=None, help='override args for bg field')

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # we will trace the background field here
        S, V, P = sizes
        fullsize = S * V * P
        
        vox_colors = fill_in((fullsize, 3), hits, all_results['colors'], 0.0)
        vox_missed = fill_in((fullsize, ), hits, all_results['missed'], 1.0)
        vox_depths = fill_in((fullsize, ), hits, all_results['depths'], 0.0)
        
        mid_dis = (self.args.near + self.args.far) / 2
        n_depth = fill_in((fullsize, ), hits, all_results['min_depths'], mid_dis)[:, None]
        f_depth = fill_in((fullsize, ), hits, all_results['max_depths'], mid_dis)[:, None]
        
        # front field
        nerf_step = getattr(self.args, "nerf_steps", 64)
        max_depth = n_depth
        min_depth = torch.ones_like(max_depth) * self.args.near
        intersection_outputs = {
            "min_depth": min_depth, "max_depth": max_depth,
            "probs": torch.ones_like(max_depth), 
            "steps": torch.ones_like(max_depth).squeeze(-1) * nerf_step,
            "intersected_voxel_idx": torch.zeros_like(min_depth).int()}
        with with_torch_seed(self.unique_seed):
            fg_samples = self.bg_encoder.ray_sample(intersection_outputs)
        fg_results = self.raymarcher(
            self.bg_encoder, self.bg_field, ray_start, ray_dir, fg_samples, {})
        
        # back field
        min_depth = f_depth
        max_depth = torch.ones_like(min_depth) * self.args.far
        intersection_outputs = {
            "min_depth": min_depth, "max_depth": max_depth,
            "probs": torch.ones_like(max_depth), 
            "steps": torch.ones_like(max_depth).squeeze(-1) * nerf_step,
            "intersected_voxel_idx": torch.zeros_like(min_depth).int()}
        with with_torch_seed(self.unique_seed):
            bg_samples = self.bg_encoder.ray_sample(intersection_outputs)
        bg_results = self.raymarcher(
            self.bg_encoder, self.bg_field, ray_start, ray_dir, bg_samples, {})

        # merge background to foreground
        all_results['voxcolors'] = vox_colors.view(S, V, P, 3)
        all_results['colors'] = fg_results['colors'] + fg_results['missed'][:, None] * (vox_colors + vox_missed[:, None] * bg_results['colors'])
        all_results['depths'] = fg_results['depths'] + fg_results['missed'] * (vox_depths + vox_missed * bg_results['depths'])
        all_results['missed'] = fg_results['missed'] * vox_missed * bg_results['missed']

        # apply the NSVF post-processing
        return super().postprocessing(ray_start, ray_dir, all_results, hits, sizes)

    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        images = super()._visualize(images, sample, output, state, **kwargs)
        if 'voxcolors' in output and output['voxcolors'] is not None:
            images['{}_vcolors/{}:HWC'.format(name, img_id)] ={
                'img': output['voxcolors'][shape, view],
                'min_val': float(self.args.min_color)
            }
        return images

    
@register_model_architecture("nsvf_bg", "nsvf_bg")
def base_bg_architecture(args):
    base_architecture(args)

@register_model_architecture("nsvf_bg", "nsvf_bg_xyz")
def base_bg2_architecture(args):
    args.nerf_steps = getattr(args, "nerf_steps", 64)
    nerf_style_architecture(args)


@register_model('shared_nsvf_bg')
class SharedNSVFBGModel(NSVFBGModel):
    
    ENCODER = 'shared_sparsevoxel_encoder'

    def postprocessing(self, ray_start, ray_dir, all_results, hits, sizes):
        # we will trace the background field here
        # pass context vector from NSVF to NeRF
        self.bg_encoder.precompute(context=self.encoder.contexts(self.encoder.cid).unsqueeze(0))
        return super().postprocessing(ray_start, ray_dir, all_results, hits, sizes)

    @torch.no_grad()
    def split_voxels(self):
        logger.info("half the global voxel size {:.4f} -> {:.4f}".format(
            self.encoder.all_voxels[0].voxel_size.item(), 
            self.encoder.all_voxels[0].voxel_size.item() * .5))
        self.encoder.splitting()
        for id in range(len(self.encoder.all_voxels)):
            self.encoder.all_voxels[id].voxel_size *= .5
            self.encoder.all_voxels[id].max_hits *= 1.5
        self.clean_caches()

    @torch.no_grad()
    def reduce_stepsize(self):
        logger.info("reduce the raymarching step size {:.4f} -> {:.4f}".format(
            self.encoder.all_voxels[0].step_size.item(), 
            self.encoder.all_voxels[0].step_size.item() * .5))
        for id in range(len(self.encoder.all_voxels)):
            self.encoder.all_voxels[id].step_size *= .5


@register_model_architecture("shared_nsvf_bg", "shared_nsvf_bg_xyz")
def base_shared_architecture(args):
    args.context_embed_dim = getattr(args, "context_embed_dim", 96)
    args.hypernetwork = getattr(args, "hypernetwork", False)
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10, context:0:96")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4:3:b")
    args.bg_field_args = getattr(args, "bg_field_args", 
        "{'inputs_to_density': 'pos:10, context:0:96', 'inputs_to_texture': 'feat:0:256, ray:4:3:b}'}")
    nerf_style_architecture(args)