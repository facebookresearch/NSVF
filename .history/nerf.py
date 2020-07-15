# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# this is an re-implementation of the original NeRF codebase
# there are some small difference in the implementation.

import logging
logger = logging.getLogger(__name__)

import cv2, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairnr.models.fairnr_model import BaseModel
from fairnr.modules.encoder import Encoder
from fairnr.modules.field import RaidanceField
from fairnr.modules.renderer import VolumeRenderer
from fairnr.modules.reader import Reader


# NeRF style volume
class VolumeEncoder(Encoder):
    """
    simple bounding volume encoder. no voxels are used.
    if no bounding box is provided, we assume a unit cube centered at origin.
    """
    def __init__(self, args, bbox_path=None):
        super().__init__(args)
        self.bbox_path = bbox_path if bbox_path is not None else args.initial_boundingbox
        self.fixed_interval = self.args.sample_fixed_interval
        if self.bbox_path is None:
            self.register_buffer("bbox", torch.tensor([[-1., -1., -1.], [1., 1., 1.]]))

    @staticmethod
    def add_args(parser):
        parser.add_argument('--initial-boundingbox', type=str, 
            help='the initial bounding box to initialize the model')
        parser.add_argument('--sample-fixed-interval', type=float, 
            help='we can sample points either based on a fixed number of samples or sample through the boundingbox')
    
    def ray_intersect(self, ray_start, ray_dir):
        S, V, P, _ = ray_dir.size()

        # ray<->bounding box intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3)
        ray_dir = ray_dir.reshape(S, V * P, 3)

        # AABB-test
        inv_ray_dir = 1. / (ray_dir + 1e-8)
        f_min, f_max = 0., 100.
        fL = (self.bbox[0][None, None, :] - ray_start) * inv_ray_dir
        fH = (self.bbox[1][None, None, :] - ray_start) * inv_ray_dir
        fmL, fmH = torch.min(fL, fH), torch.max(fL, fH)
        fmL, fmH = fmL.max(dim=2)[0], fmH.min(dim=2)[0]
        missed = (fmL > fmH) | (fmH < f_min) | (fmL > f_max)

        hits = ~missed
        min_depth = fmL.clamp(min=f_min)
        max_depth = fmH.clamp(max=f_max)
        
        # get the pts_index
        pts_idx = hits.long() - 1
        pts_idx = (pts_idx + torch.arange(S, 
                    device=pts_idx.device, dtype=pts_idx.dtype)[:, None]
                    ).masked_fill_(pts_idx.eq(-1), -1).unsqueeze(-1)
        return ray_start, ray_dir, min_depth, max_depth, pts_idx, hits

    def ray_sample(self, pts_idx, min_depth, max_depth):
        raise NotImplementedError("NeRF volume encoder is not implemented yet.")


@register_model('nerf')
class NeRFModel(BaseModel):

    @classmethod
    def build_encoder(cls, args):
        return VolumeEncoder(args)

    @classmethod
    def build_field(cls, args):
        return RaidanceField(args, 3)

    @classmethod
    def build_raymarcher(cls, args):
        return VolumeRenderer(args)

    @staticmethod
    def add_args(parser):
        VolumeEncoder.add_args(parser)
        RaidanceField.add_args(parser)
        VolumeRenderer.add_args(parser)
        Reader.add_args(parser)

    def _forward(self, ray_start, ray_dir, **kwargs):
        BG_DEPTH = self.field.bg_color.depth

        raise NotImplementedError("NeRF is not implemented.")

        S, V, P, _ = ray_dir.size()
        fullsize = S * V * P
        assert S == 1, "naive NeRF only supports single object."

        # ray-boudning box intersection (in default unit cube)
        ray_start, ray_dir, min_depth, max_depth, pts_idx, hits = \
            self.encoder.ray_intersect(ray_start, ray_dir)

        if hits.sum() > 0:  # check if ray missed the volume
            ray_start, ray_dir = ray_start[hits], ray_dir[hits]
            pts_idx, min_depth, max_depth = pts_idx[hits], min_depth[hits], max_depth[hits]

            # sample evalution points along the ray
            samples = self.encoder.ray_sample(pts_idx, min_depth, max_depth)

            from fairseq import pdb; pdb.set_trace()


@register_model_architecture("nerf", "nerf_base")
def base_architecture(args):
    # encoder
    args.initial_boundingbox = getattr(args, "initial_boundingbox", None)
    args.sample_fixed_interval = getattr(args, "sample_fixed_interval", 4)

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