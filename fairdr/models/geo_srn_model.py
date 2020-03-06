# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairdr.data.geometry import ray
from fairdr.models.srn_model import SRNModel, SRNField, base_architecture

@register_model('geometry_aware_srn')
class GEOSRNModel(SRNModel):

    @classmethod
    def build_field(cls, args):
        return GEOSRNField(args)

    def forward(self, ray_start, ray_dir, voxels, points, raymarching_steps=None, **kwargs):
        intersection = self.field.voxel_intersect(
            self.field.voxel_travsal(
                ray_start, ray_dir),
            voxels[:, :3])
        intersection = intersection.sum(-1) > 0   # intersection map
        if self.args.pixel_per_view is None or (not self.training):
            inter_mask = self.field.dilate(intersection, k=10)  # making it larger
        else:
            inter_mask = intersection

        # masked ray-marching
        ray_start, ray_dir = ray_start.expand_as(ray_dir)[inter_mask], ray_dir[inter_mask]
        depths, _ = self.raymarcher(
            self.field.get_sdf, 
            ray_start, ray_dir, 
            steps=self.args.raymarching_steps)
        points = ray(ray_start, ray_dir, depths.unsqueeze(-1))
        predicts = self.field(points)

        # mapping to normal size
        output_shape = tuple(list(inter_mask.size()) + [3])
        predicts = (predicts.new_ones(3) * self.field.bg_color)[
            None, None, None, :].expand(*output_shape).masked_scatter(
                inter_mask.unsqueeze(-1).expand(*output_shape), predicts)
        depths = (depths.new_ones(*output_shape[:-1]) * self.field.max_depth
            ).masked_scatter(inter_mask, depths)
    
        # model's output
        results = {
            'predicts': predicts,
            'depths': depths,
            'grad_penalty': 0
        }

        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results


class GEOSRNField(SRNField):

    def __init__(self, args):
        super().__init__(args)
        self.root = [-2.5, -2.5, -2.5]
        self.voxel_size = 5 / 128
        self.valid_step = [1.5, 4.5]
        self.max_depth = 5
        self.bg_color = nn.Parameter(
            torch.tensor([0, 0, 0], dtype=torch.float32), 
            requires_grad=True)

    def point2voxel(self, xyz):
        xyz[:, 0] -= self.root[0]
        xyz[:, 1] -= self.root[1]
        xyz[:, 2] -= self.root[2]
        return (xyz / self.voxel_size).floor_().int()

    @torch.no_grad()
    def dilate(self, mask, k=4):
        
        S, V, DD = mask.size()
        D = int(math.sqrt(DD))
        mask = mask.view(S, V, D, D)
        new_mask = torch.zeros_like(mask)

        for ki in range(1, k):
            new_mask[:, :, :-ki, :] |= mask[:, :, ki:, :]
            new_mask[:, :, ki:, :] |= mask[:, :, :-ki, :]
            new_mask[:, :, :, :-ki] |= mask[:, :, :, ki:]
            new_mask[:, :, :, ki:] |= mask[:, :, :, :-ki]
        
        return new_mask.view(S, V, DD)

    @torch.no_grad()
    def point_range(self, model_points):
        return model_points.norm(dim=1, p=2).max()

    @torch.no_grad()
    def voxel_intersect(self, ray_voxels, model_voxels):
        # HACK: this is a hack, better looking for better and more robust options.
        # crop the input voxel to relatively save memory
        
        def crop(voxels, min, max):
            voxels[:, 0] = voxels[:, 0].clamp(min[0], max[0]) - min[0]
            voxels[:, 1] = voxels[:, 1].clamp(min[1], max[1]) - min[1]
            voxels[:, 2] = voxels[:, 2].clamp(min[2], max[2]) - min[2]
            return voxels

        def flatten(voxels, shape):
            return (voxels[:, 0] * shape[1] * shape[2] + voxels[:, 1] * shape[2] + voxels[:, 2]).long()

        def build_dense(voxels, shape):
            return voxels.new_zeros(*shape).view(-1).scatter_(0, flatten(voxels, shape), 1)

        S, V, P, D, _ = ray_voxels.size()
        
        min_voxel, max_voxel = model_voxels.min(0)[0] - 1, model_voxels.max(0)[0] + 1
        ray_voxels = crop(ray_voxels.view(-1, 3), min_voxel, max_voxel)
        dense_shape = (max_voxel - min_voxel + 1).cpu().tolist()
        dense_voxels = build_dense(crop(model_voxels, min_voxel, max_voxel), dense_shape)
        intersection = dense_voxels[flatten(ray_voxels, dense_shape)]
        return intersection.view(S, V, P, D)
        
    @torch.no_grad()
    def voxel_travsal(self, ray_start, ray_dir, valid_step=None):
        S, V, P, _ = ray_dir.size()
        if valid_step is None:
            valid_step = self.valid_step
        t0, t1 = valid_step
        depths = torch.arange(t0, t1, self.voxel_size, 
                    device=ray_dir.device, dtype=ray_dir.dtype)
        D = depths.size(0)
        points = ray_start[:, :, :, None, :] \
               + ray_dir[:, :, :, None, :] * depths[None, None, None, :, None]
        voxels = self.point2voxel(points.view(-1, 3)).view(S, V, P, D, 3)
        return voxels
        

@register_model_architecture("geometry_aware_srn", "geosrn_simple")
def geo_base_architecture(args):
    base_architecture(args)
