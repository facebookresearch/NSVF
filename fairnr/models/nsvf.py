# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from plyfile import PlyData, PlyElement

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairnr.modules.pointnet2.pointnet2_utils import (
    aabb_ray_intersect, 
    uniform_ray_sampling
)
from fairnr.data.geometry import (
    ray, trilinear_interp, get_edge,
    pruning_points, offset_points,
    compute_normal_map
)
from fairnr.models.fairnr_model import BaseModel
from fairnr.modules.encoder import SparseVoxelEncoder
from fairnr.modules.field import RaidanceField
from fairnr.modules.renderer import VolumeRenderer
from fairnr.modules.reader import Reader
from fairnr.modules.linear import Linear, NeRFPosEmbLinear, Embedding
from fairnr.modules.implicit import (
    ImplicitField, SignedDistanceField, TextureField, DiffusionSpecularField,
    HyperImplicitField, SphereTextureField
)

MAX_DEPTH = 10000.0


@register_model('nsvf')
class NSVFModel(BaseModel):

    @classmethod
    def build_encoder(cls, args):
        return SparseVoxelEncoder(args)

    @classmethod
    def build_field(cls, args):
        return RaidanceField(args, args.voxel_embed_dim)

    @classmethod
    def build_raymarcher(cls, args):
        return VolumeRenderer(args)

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        RaidanceField.add_args(parser)
        VolumeRenderer.add_args(parser)
        Reader.add_args(parser)

    def _forward(self, ray_start, ray_dir, **kwargs):
        BG_DEPTH = self.field.bg_color.depth

        # initialization (typically S == 1)
        S, V, P, _ = ray_dir.size()
        fullsize = S * V * P

        # voxel encoder (precompute for each voxel if needed)
        feats, xyz, values = self.encoder.precompute(**kwargs)  # feats: (S, B, 8), xyz: (S, B, 3),  values: (S, B', D)

        # ray-voxel intersection
        ray_start, ray_dir, min_depth, max_depth, pts_idx, hits = \
            self.encoder.ray_voxel_intersect(ray_start, ray_dir, xyz, feats)

        if hits.sum() > 0:  # check if ray missed everything
            ray_start, ray_dir = ray_start[hits], ray_dir[hits]
            pts_idx, min_depth, max_depth = pts_idx[hits], min_depth[hits], max_depth[hits]
            
            # sample evalution points along the ray
            samples = self.encoder.ray_sample(pts_idx, min_depth, max_depth)
            
            # volume rendering
            encoder_states = (feats.reshape(-1, 8), xyz.reshape(-1, 3), values.reshape(-1, values.size(-1)))
            colors, depths, missed, var_loss = self.raymarcher(
                self.encoder, self.field, ray_start, ray_dir, samples, encoder_states)
            depths = depths + BG_DEPTH * missed
            voxel_edges = get_edge(
                ray_start + ray_dir * samples[0][:, :1], 
                xyz.reshape(-1, 3)[samples[1][:, 0].long()], 
                self.encoder.voxel_size).type_as(depths)   # get voxel edges/depth (for visualization)
            voxel_edges = (1 - voxel_edges[:, None].expand(voxel_edges.size(0), 3)) * 0.7
            voxel_depth = samples[0][:, 0]
        
        else:
            colors, depths, missed, voxel_edges, voxel_depth = None, None, None, None, None
            var_loss = torch.tensor(0.0).type_as(depths)

        # fill-in
        def fill_in(shape, hits, input, initial=1.0):
            output = ray_dir.new_ones(*shape) * initial
            if input is not None:
                if len(shape) == 1:
                    return output.masked_scatter(hits, input)
                return output.masked_scatter(hits.unsqueeze(-1).expand(*shape), input)

        hits = hits.reshape(fullsize)
        missed = fill_in((fullsize,), hits, missed, 1.0)
        depths = fill_in((fullsize,), hits, depths, BG_DEPTH)
        voxel_depth = fill_in((fullsize,), hits, voxel_depth, BG_DEPTH)
        voxel_edges = fill_in((fullsize, 3), hits, voxel_edges, 1.0)
        bg_color = self.field.bg_color(**kwargs)
        colors = fill_in((fullsize, 3), hits, colors, 0.0) + missed.unsqueeze(-1) * bg_color.reshape(-1, 3)
        
        # model's output
        return {
            'colors': colors.view(S, V, P, 3),
            'depths': depths.view(S, V, P),
            'missed': missed.view(S, V, P),
            'voxel_edges': voxel_edges.view(S, V, P, 3),
            'voxel_depth': voxel_depth.view(S, V, P),
            'var_loss': var_loss,
            'bg_color': bg_color,
            'other_logs': {
                'voxs_log': self.encoder.voxel_size.item(),
                'stps_log': self.encoder.step_size.item(),
                }
        }

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
        return images
    
    @torch.no_grad()
    def prune_voxels(self, th=0.5):
        self.encoder.pruning(self.field, th)
    
    @torch.no_grad()
    def split_voxels(self):
        logger.info("half the global voxel size {:.4f} -> {:.4f}".format(
            self.encoder.voxel_size.item(), self.encoder.voxel_size.item() * .5))
        self.encoder.splitting()
        self.encoder.voxel_size *= .5
        self.encoder.max_hits *= 1.5
        
    @torch.no_grad()
    def reduce_stepsize(self):
        logger.info("reduce the raymarching step size {:.4f} -> {:.4f}".format(
            self.encoder.step_size.item(), self.encoder.step_size.item() * .5))
        self.encoder.step_size *= .5


@register_model_architecture("nsvf", "nsvf_base")
def base_architecture(args):
    # encoder
    args.voxel_size = getattr(args, "voxel_size", 0.25)
    args.voxel_path = getattr(args, "voxel_path", None)
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 32)
    args.total_num_embedding = getattr(args, "total_num_embedding", None)
    args.initial_boundingbox = getattr(args, "initial_boundingbox", None)
    args.xyz_as_voxel_embed = getattr(args, "xyz_as_voxel_embed", False)

    # field
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)
    args.input_embed_dim = getattr(args, "input_embed_dim", 3)
    args.output_embed_dim = getattr(args, "output_embed_dim", 256)
    args.raydir_embed_dim = getattr(args, "raydir_embed_dim", 24)
    args.disable_raydir = getattr(args, "disable_raydir", False)
    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    args.add_pos_embed = getattr(args, "add_pos_embed", 6)
    args.saperate_specular = getattr(args, "saperate_specular", False)
    args.specular_dropout = getattr(args, "specular_dropout", 0.0)
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.background_depth = getattr(args, "background_depth", 5.0)
    
    # raymarcher
    args.max_hits = getattr(args, "max_hits", 60)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
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
    args.chunk_size = getattr(args, "chunk_size", 256)

@register_model_architecture("nsvf", "nsvf_xyz")
def nerf_architecture(args):
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 3)
    args.add_pos_embed = getattr(args, "add_pos_embed", 10)
    args.xyz_as_voxel_embed = getattr(args, "xyz_as_voxel_embed", True)
    base_architecture(args)


