# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairdr.modules.pointnet2.pointnet2_utils import ball_ray_intersect, aabb_ray_intersect
from fairdr.data.geometry import ray
from fairdr.models.point_srn_model import (
    PointSRNModel, PointSRNField, 
    transformer_base_architecture,
    pointnet_base_architecture,
    embedding_base_architecture
)
from fairdr.modules.raymarcher import BG_DEPTH, MAX_DEPTH


@register_model('dev_srn')
class GEOSRNModel(PointSRNModel):

    @classmethod
    def build_field(cls, args):
        return GEOSRNField(args)

    @staticmethod
    def add_args(parser):
        PointSRNModel.add_args(parser)
        parser.add_argument('--max-hits', type=int, metavar='N',
                            help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--bounded', action='store_true', 
                            help='ray will be either bounded in the ball or missed in BG_DEPTH')
        parser.add_argument('--sdf-scale', type=float, metavar='D')
        parser.add_argument('--march-from-center', action='store_true')
        parser.add_argument('--march-with-ball', action='store_true')
        parser.add_argument('--background-feature', action='store_true')
        parser.add_argument('--intersection-type', choices=['ball', 'aabb'], default='ball')

    def forward(self, ray_start, ray_dir, points, raymarching_steps=None, **kwargs):
        # get geometry features
        S, V, P, _ = ray_dir.size()
        feats, xyz = self.field.get_backbone_features(points)
        
        # xyz1 = xyz.clone(); xyz1[:, :, 0] += 0.2; feats1 = feats[xyz1[:,:,2]>0].unsqueeze(0); xyz1 = xyz1[xyz1[:,:,2]>0].unsqueeze(0)
        # xyz2 = xyz.clone(); xyz2[:, :, 0] -= 0.2; feats2 = feats[xyz2[:,:,2]<0].unsqueeze(0); xyz2 = xyz2[xyz2[:,:,2]<0].unsqueeze(0)
        # xyz = torch.cat([xyz1, xyz2], 1)
        # feats = torch.cat([feats1, feats2], 1)

        # coarse ray-intersection
        hit_idx, _ray_start, _ray_dir, state, min_depth, max_depth = \
            self.field.ray_intersection(ray_start, ray_dir, xyz, feats)
        start_depth, min_depth, max_depth = self.field.adjust_depth(min_depth, max_depth)

        # fine-grained ray-intersection
        depths, _ = self.raymarcher(
            self.field.get_sdf, 
            _ray_start, _ray_dir,
            state=state,
            steps=self.args.raymarching_steps 
                if raymarching_steps is None else raymarching_steps,
            min=start_depth if self.args.bounded else 0.0,
            max=None)
        missed = abs(depths - start_depth) - ((max_depth - min_depth) / 2.0)
        hit_idx, pts_idx, depths, missed, _points, _feats, _xyz = self.field.prepare_for_rendering(
            hit_idx, depths, missed, feats, xyz, ray_start, ray_dir)
        
        # only render "no-background colors"
        _predicts = self.field(_points, _feats, _xyz)
        predicts = self.field.bg_color.unsqueeze(0).expand(S * V * P, 3)
        predicts = predicts.masked_scatter(
            hit_idx.unsqueeze(1).expand(S * V * P, 3),
            _predicts)
        
        # model's output
        results = {
            'predicts': predicts.view(S, V, P, 3),
            'depths': depths.view(S, V, P),
            'missed': missed.view(S, V, P),
            'grad_penalty': 0,
            'hits': pts_idx.view(S, V, P),
            'min_depths': 0.0
        }
    
        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results
    

class GEOSRNField(PointSRNField):

    def __init__(self, args):
        super().__init__(args)
        self.max_hits = args.max_hits
        self.ball_radius = args.ball_radius
        self.bg_color = nn.Parameter(
            torch.tensor((1.0, 1.0, 1.0)) * getattr(args, "transparent_background", -0.8), 
            requires_grad=(not getattr(args, "no_background_loss", False)))
        self.sdf_scale = getattr(args, 'sdf_scale', 0.1)
        self.bg_feature = nn.Parameter(
            torch.normal(0, 0.02, (self.backbone.feature_dim, ))
        ) if getattr(args, "background_feature", False) else None
        self.march_with_ball = getattr(args, "march_with_ball", False)
        self.intersection_type = getattr(args, "intersection_type", 'ball')

    def adjust_depth(self, min_depth, max_depth):
        start_depth = (min_depth + max_depth) / 2
        if not self.march_with_ball:
            if self.intersection_type == 'aabb':
                radius = self.ball_radius * 0.866
            else:
                radius = self.ball_radius
            min_depth = start_depth - radius
            max_depth = start_depth + radius
        return start_depth.detach(), min_depth.detach(), max_depth.detach()

    def prepare_for_rendering(self, hit_idx, depths, missed, point_feats, point_xyz, ray_start, ray_dir):
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        depths = depths.masked_scatter(
            missed > 0, torch.zeros_like(missed > 0).float().uniform_() + (MAX_DEPTH - 2.0))  # add random
        depth_map = torch.ones_like(hit_idx).float() * MAX_DEPTH
        depth_map = depth_map.masked_scatter(hit_idx.ne(-1), depths)
        depth_map, _idx = depth_map.min(dim=-1)   # select the nearest point

        missd_map = torch.ones_like(hit_idx).float()
        missd_map = missd_map.masked_scatter(hit_idx.ne(-1), missed)
        missd_map = missd_map.gather(1, _idx.unsqueeze(1)).squeeze(1)
        pts_idx = hit_idx.gather(1, _idx.unsqueeze(1)).squeeze(1)
        
        hit_idx = missd_map < 0
        depth_map = depth_map.masked_fill(missd_map > 0, BG_DEPTH)
        point_feats = point_feats.view(S * H, D)[pts_idx[hit_idx]]
        point_xyz = point_xyz.view(S * H, 3)[pts_idx[hit_idx]]
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S * V * P, 3)[hit_idx]
        ray_dir = ray_dir.view(S * V * P, 3)[hit_idx]
        depths = depth_map[hit_idx]

        points = ray(ray_start, ray_dir, depths.unsqueeze(-1))
        return hit_idx, pts_idx, depth_map, missd_map, points, point_feats, point_xyz

    def ray_intersection(self, ray_start, ray_dir, point_xyz, point_feats):
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        ray_idx = torch.arange(S * V * P, device=ray_start.device).long()
        ray_idx = ray_idx.unsqueeze(-1).expand(S * V * P, self.max_hits)
        if self.intersection_type == 'ball':
            ray_intersect_fn = ball_ray_intersect 
        elif self.intersection_type == 'aabb':
            ray_intersect_fn = aabb_ray_intersect
        else:
            raise NotImplementedError
        
        pts_idx, min_depth, max_depth = ray_intersect_fn(
            self.ball_radius, self.max_hits, point_xyz, 
            ray_start.expand_as(
                ray_dir).contiguous().view(S, V * P, 3), 
            ray_dir.view(S, V * P, 3))
        # print(pts_idx.ne(-1).float().sum(-1).max().item())

        hit_idx = pts_idx.view(S * V * P, self.max_hits).long()
        pts_idx = (pts_idx + H * torch.arange(S, device=pts_idx.device)[:, None, None])
        pts_idx = pts_idx.view(S * V * P, self.max_hits)[hit_idx.ne(-1)]
        ray_idx = ray_idx[hit_idx.ne(-1)]
        
        min_depth = min_depth.view(S * V * P, self.max_hits)[hit_idx.ne(-1)]
        max_depth = max_depth.view(S * V * P, self.max_hits)[hit_idx.ne(-1)]
        
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S * V * P, 3)[ray_idx]
        ray_dir = ray_dir.view(S * V * P, 3)[ray_idx]
        point_feats = point_feats.view(S * H, D)[pts_idx]
        point_xyz = point_xyz.view(S * H, 3)[pts_idx]

        return hit_idx, ray_start, ray_dir, (point_feats, point_xyz, None), min_depth, max_depth

    def get_feature(self, xyz, point_feats, point_xyz):
        # assert self.relative_position, "we only support relative position for now."
        if self.relative_position:
            input_feats = torch.cat([point_feats, self.point_proj(xyz - point_xyz)], -1)
        else:
            input_feats = torch.cat([point_feats, self.point_proj(xyz)], -1)
        return self.feature_field(input_feats)

    def get_sdf(self, xyz, state=None):
        point_feats, point_xyz, hidden_state = state
        output_feature = self.get_feature(xyz, point_feats, point_xyz)
        depth, hidden_state = self.signed_distance_field(output_feature, hidden_state)
        depth = depth * self.sdf_scale
        return depth, (point_feats, point_xyz, hidden_state)

    def get_texture(self, xyz, point_feats, point_xyz, dir=None):
        features = self.get_feature(xyz, point_feats, point_xyz)
        if dir is not None and self.use_raydir:
            features = torch.cat([features, self.raydir_proj(dir)], -1)
        return self.renderer(features)

@register_model_architecture("dev_srn", "dev_srn1")
def geo_base_architecture(args):
    args.max_hits = getattr(args, "max_hits", 20)
    args.ball_radius = getattr(args, "ball_radius", 0.08)
    args.lstm_sdf = getattr(args, "lstm_sdf", False)
    args.transformer_input_shuffle = getattr(args, "transformer_input_shuffle", True)
    args.bounded = getattr(args, "bounded", True)
    args.sdf_scale = getattr(args, "sdf_scale", 0.1)
    args.march_from_center = getattr(args, "march_from_center", True)
    args.march_with_ball = getattr(args, "march_with_ball", False)
    args.background_feature = getattr(args, "background_feature", False)
    transformer_base_architecture(args)

@register_model_architecture("dev_srn", "dev_srn2")
def geo_base2_architecture(args):
    args.march_from_center = getattr(args, "march_from_center", True)
    args.march_with_ball = getattr(args, "march_with_ball", False)
    args.background_feature = getattr(args, "background_feature", False)
    args.max_hits = getattr(args, "max_hits", 20)
    args.ball_radius = getattr(args, "ball_radius", 0.1)
    args.lstm_sdf = getattr(args, "lstm_sdf", False)
    args.pointnet2_input_shuffle = getattr(args, "pointnet2_input_shuffle", True)
    args.bounded = getattr(args, 'bounded', True)
    args.sdf_scale = getattr(args, "sdf_scale", 0.1)
    pointnet_base_architecture(args)
    
@register_model_architecture("dev_srn", "dev_srn3")
def geo_base3_architecture(args):
    args.max_hits = getattr(args, "max_hits", 20)
    args.ball_radius = getattr(args, "ball_radius", 0.08)
    args.lstm_sdf = getattr(args, "lstm_sdf", False)
    args.bounded = getattr(args, "bounded", True)
    args.sdf_scale = getattr(args, "sdf_scale", 0.1)
    args.march_from_center = getattr(args, "march_from_center", True)
    args.march_with_ball = getattr(args, "march_with_ball", False)
    args.background_feature = getattr(args, "background_feature", False)
    args.quantized_input_shuffle = getattr(args, "quantized_input_shuffle", True)
    embedding_base_architecture(args)