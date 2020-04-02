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
from fairdr.modules.pointnet2.pointnet2_utils import aabb_ray_intersect, uniform_ray_sampling
from fairdr.data.geometry import ray
from fairdr.models.point_srn_model import (
    PointSRNModel, PointSRNField, 
    embedding_base_architecture
)
from fairdr.models.fairdr_model import Raymarcher
from fairdr.modules.raymarcher import BG_DEPTH, MAX_DEPTH


@register_model('geo_nerf')
class GEONERFModel(PointSRNModel):

    @classmethod
    def build_field(cls, args):
        return GEORadianceField(args)

    @classmethod
    def build_raymarcher(cls, args):
        return GEORadianceRenderer(args)

    @staticmethod
    def add_args(parser):
        PointSRNModel.add_args(parser)
        parser.add_argument('--max-hits', type=int, metavar='N',
                            help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--voxel-size', type=float, metavar='D',
                            help='voxel size of the input points')
        parser.add_argument('--chunk-size', type=int, metavar='D', 
                            help='set chunks to go through the network. trade time for memory')
        parser.add_argument('--outer-chunk-size', type=int, metavar='D',
                            help='chunk the input rays in the very beginning if the image was too large.')
        parser.add_argument('--inner-chunking', action='store_true', 
                            help="if set, chunking in the field function, otherwise, chunking in the rays")
        
        parser.add_argument('--raymarching-stepsize', type=float, metavar='N',
                            help='ray marching step size')
        parser.add_argument('--bounded', action='store_true', 
                            help='ray will be either bounded in the ball or missed in BG_DEPTH')
        parser.add_argument('--background-stop-gradient', action='store_true')
        parser.add_argument('--intersection-type', choices=['ball', 'aabb'], default='aabb')
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--sigmoid-activation', action='store_true',
                            help='if set, instead of using the default steps. we use sigmoid function')
        parser.add_argument('--expectation', choices=['rgb', 'depth', 'features'], type=str,
                            help='where to compute the expectation from, in default is rgb.')


    def _forward(self, ray_start, ray_dir, feats, xyz, **kwargs):
        # coarse ray-intersection
        S, V, P, _ = ray_dir.size()
        predicts = self.field.bg_color.unsqueeze(0).expand(S * V * P, 3)
        depths = predicts.new_ones(S * V * P) * BG_DEPTH

        hits, _ray_start, _ray_dir, state, samples = \
            self.field.ray_intersection(ray_start, ray_dir, xyz, feats) 
        if hits.sum() <= 0:   # missed everything
            return predicts.view(S, V, P, 3), depths.view(S, V, P)
        hits = hits.view(S * V * P).contiguous()

        # fine-grained raymarching + rendering
        _predicts, _depths, _missed = self.raymarcher(self.field, _ray_start, _ray_dir, samples, state)
        _predicts = _predicts + self.field.bg_color * _missed.unsqueeze(-1)  # fill background color
        _depths = _depths + BG_DEPTH * _missed

        predicts = predicts.masked_scatter(
            hits.unsqueeze(-1).expand(S * V * P, 3),
            _predicts)
        depths = depths.masked_scatter(
            hits, _depths
        )
        return predicts.view(S, V, P, 3), depths.view(S, V, P)

    def forward(self, ray_start, ray_dir, points, raymarching_steps=None, **kwargs):
        # get geometry features
        feats, xyz = self.field.get_backbone_features(points)

        # from fairseq import pdb; pdb.set_trace()
        chunk_size = 400 * 400 # getattr(self.args, "outer_chunk_size", 400 * 400)
        if chunk_size >= ray_dir.size(2):
            # no need to chunk if the input is small
            predicts, depths = self._forward(ray_start, ray_dir, feats, xyz, **kwargs)
        
        else:            
            predicts, depths = zip(*[
                self._forward(ray_start, ray_dir[:, :, i: i+chunk_size], feats, xyz, **kwargs)
            for i in range(0, ray_dir.size(2), chunk_size)])
            predicts, depths = torch.cat(predicts, 2), torch.cat(depths, 2)

        # model's output
        results = {
            'predicts': predicts,
            'depths': depths, 
        }
    
        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results


class GEORadianceField(PointSRNField):
    
    def __init__(self, args):
        super().__init__(args)
        self.max_hits = args.max_hits
        self.step_size = args.raymarching_stepsize
        self.chunk_size = 1024 * args.chunk_size
        self.bg_color = nn.Parameter(
            torch.tensor((1.0, 1.0, 1.0)) * getattr(args, "transparent_background", -0.8), 
            requires_grad=(not getattr(args, "background_stop_gradient", False)))
        self.intersection_type = getattr(args, "intersection_type", 'aabb')
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.inner_chunking = getattr(args, "inner_chunking", True)
        self.trilinear_interp = False
        if args.backbone == 'embedding':
            self.trilinear_interp = getattr(args, "quantized_voxel_vertex", False)

        assert not args.lstm_sdf, "we do not support LSTM field"
        assert self.intersection_type == 'aabb', "we only support AABB for now"

    def get_backbone_features(self, points):
        if not self.trilinear_interp:
           return self.backbone(points, add_dummy=True)

        S, H, _ = points.size()
        feats, _ = self.backbone((points[:, :, None, :] + 
            self.backbone.offset[None, None, :, :]).view(S, H * 8, 3))
        return feats.view(S, H, -1), points

    def get_feature(self, xyz, point_feats, point_xyz):
        point_feats = self.backbone.get_features(point_feats).view(point_feats.size(0), -1)
        if self.trilinear_interp:
            xyz = (xyz - point_xyz) / self.ball_radius + 0.5
            xyz = xyz[:, None, :]
            offset = self.backbone.offset / self.ball_radius + 0.5
            offset = offset[None, :, :]
            weights = (xyz * offset + (1 - xyz) * (1 - offset)).prod(dim=-1, keepdim=True) 
            point_feats = (weights * point_feats.view(point_feats.size(0), 8, -1)).sum(1)
            xyz = xyz.squeeze(1)

        if self.raypos_features > 0:
            if self.relative_position:
                point_feats = torch.cat([point_feats, self.point_proj(xyz - point_xyz)], -1)
            else:
                point_feats = torch.cat([point_feats, self.point_proj(xyz)], -1)
        return self.feature_field(point_feats)

    def _forward(self, xyz, point_feats, point_xyz, dir=None, features=None, outputs=['sigma', 'texture']):
        _data = {}
        if features is None:
            features = self.get_feature(xyz, point_feats, point_xyz)
            _data['features'] = features

        if 'sigma' in outputs:
            sigma = self.signed_distance_field(features)[0]
            _data['sigma'] = sigma

        if 'texture' in outputs:
            if dir is not None and self.use_raydir:
                features = torch.cat([features, self.raydir_proj(dir)], -1)
            texture = self.renderer(features)
            _data['texture'] = texture
        return tuple([_data[key] for key in outputs])

    def forward(self, *args, **kwargs):
        if not self.inner_chunking:
            return self._forward(*args, **kwargs)

        chunk_size = self.chunk_size
        outputs = zip(*[
                self._forward(*[a[i: i + chunk_size] 
                                if isinstance(a, torch.Tensor) else a 
                                for a in args], **kwargs) 
            for i in range(0, args[-1].shape[0], chunk_size)])
        
        outputs = [torch.cat(o, 0) 
                    if isinstance(o[0], torch.Tensor) 
                    else o for o in outputs]
        return tuple(outputs)

    def ray_intersection(self, ray_start, ray_dir, point_xyz, point_feats):
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3)
        ray_dir = ray_dir.reshape(S, V * P, 3)
        ray_intersect_fn = aabb_ray_intersect
        pts_idx, min_depth, max_depth = ray_intersect_fn(
            self.ball_radius, self.max_hits, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        pts_idx = (pts_idx + H * torch.arange(S, 
            device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None])
        
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        if hits.sum() <= 0:            # missed everything
            return hits, None, None, None, None
        
        pts_idx = pts_idx[hits]
        min_depth = min_depth[hits]
        max_depth = max_depth[hits]
        max_steps = int(((max_depth - min_depth).sum(-1) / self.step_size).ceil_().max())

        sampled_idx, sampled_depth, sampled_dists = [p.squeeze(0) for p in 
            uniform_ray_sampling(
                self.step_size, max_steps, 
                pts_idx.unsqueeze(0), min_depth.unsqueeze(0), max_depth.unsqueeze(0),
                self.deterministic_step or (not self.training)
            )]
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        
        # prepare output
        ray_start = ray_start[hits]
        ray_dir = ray_dir[hits]

        point_feats = point_feats.view(S * H, D)
        point_xyz = point_xyz.view(S * H, 3)

        return hits, ray_start, ray_dir, (point_feats, point_xyz), (sampled_depth, sampled_idx, sampled_dists)


class GEORadianceRenderer(Raymarcher):

    def __init__(self, args):
        super().__init__(args)
        self.step_size = args.raymarching_stepsize
        self.chunk_size = 1024 * args.chunk_size
        self.inner_chunking = getattr(args, "inner_chunking", True)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.sigmoid_activation = getattr(args, "sigmoid_activation", False)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.expectation = getattr(args, "expectation", "rgb")

    def _forward(self, field_fn, ray_start, ray_dir, samples, state=None):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        point_feats, point_xyz = state
        sampled_depth, sampled_idx, sampled_dists = samples
        sampled_idx = sampled_idx.long()

        H, D = point_feats.size()
        P, M = sampled_idx.size()

        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        queries = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        queries = queries[sample_mask]
        querie_dirs = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])[sample_mask]
        sampled_idx = sampled_idx[sample_mask]
        sampled_dists = sampled_dists[sample_mask]

        point_xyz = point_xyz.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), 3))
        point_feats = point_feats.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), D))

        # get sigma & rgb
        if self.expectation == 'rgb':
            sigma, texture = field_fn(queries, point_feats, point_xyz, querie_dirs)
        elif self.expectation == 'features':
            sigma, features = field_fn(queries, point_feats, point_xyz, querie_dirs, outputs=['sigma', 'features'])
        elif self.expectation == 'depth':
            sigma,  = field_fn(queries, point_feats, point_xyz, querie_dirs, outputs=['sigma'])
        else:
            raise NotImplementedError

        noise = 0 if not self.discrete_reg and (not self.training) \
            else torch.zeros_like(sigma).normal_()
        
        if not self.sigmoid_activation:
            if self.deterministic_step:
                dists = self.step_size
            else:
                dists = sampled_dists
            sigma_dist = torch.relu(noise + sigma) * dists  # follow NERF paper implementation
        
        else:
            assert self.deterministic_step, "sigmoid activation only supports deterministic steps"
            sigma_dist = -torch.log(torch.sigmoid(sigma + noise))

        sigma_dist = torch.zeros_like(sampled_depth).masked_scatter(sample_mask, sigma_dist)
        shifted_sigma_dist = torch.cat([sigma_dist.new_zeros(P, 1), sigma_dist[:, :-1]], dim=-1)  # shift one step
        probs = ((1 - torch.exp(-sigma_dist.float())) * torch.exp(-torch.cumsum(shifted_sigma_dist.float(), dim=-1)))
        probs = probs.type_as(sigma_dist)
        
        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)

        if self.expectation == 'rgb':
            texture = sigma_dist.new_zeros(*sigma_dist.size(), 3).masked_scatter(
                sample_mask.unsqueeze(-1).expand(*sample_mask.size(), 3), texture)
            rgb = (texture * probs.unsqueeze(-1)).sum(-2)

        elif self.expectation == 'features':
            features = sigma_dist.new_zeros(*sigma_dist.size(), features.size(-1)).masked_scatter(
                sample_mask.unsqueeze(-1).expand(*sample_mask.size(), features.size(-1)), features).float()
            probs = probs.float()
            features = (features * (probs / (1e-6 + probs.sum(-1, keepdim=True))).unsqueeze(-1)).sum(-2).type_as(ray_dir)
            texture, = field_fn(None, None, None, ray_dir, features, outputs=['texture'])
            rgb = (texture * (1 - missed).unsqueeze(-1))
        
        elif self.expectation == 'depth':
            exp_depth = (depth.float() / (1e-6 + probs.float().sum(-1))).type_as(depth)
            exp_index = sampled_idx[(exp_depth.unsqueeze(-1) > sampled_depth).sum(-1)]
            exp_point_xyz = state[1].gather(0, exp_index.unsqueeze(1).expand(exp_index.size(0), 3))
            exp_point_feats = state[0].gather(0, exp_index.unsqueeze(1).expand(exp_index.size(0), D))
            exp_xyz = ray(ray_start, ray_dir, exp_depth.unsqueeze(-1))
            texture, = field_fn(exp_xyz, exp_point_feats, exp_point_xyz, ray_dir, outputs=['texture'])
            rgb = (texture * (1 - missed).unsqueeze(-1))

        else:
            raise NotImplementedError

        return rgb, depth, missed

    def forward(self, field_fn, ray_start, ray_dir, samples, state=None):
        if self.inner_chunking:
            return self._forward(field_fn, ray_start, ray_dir, samples, state)

        """
        use chunks to trade time to memory (save more!!)
        """
        sampled_depth, sampled_idx, sampled_dists = samples
        num_hits = sampled_idx.ne(-1).sum(-1)
        sorted_hits, sorted_idx = num_hits.sort(dim=-1)

        ray_start, ray_dir, sampled_depth, sampled_idx, sampled_dists = \
            ray_start[sorted_idx], ray_dir[sorted_idx], sampled_depth[sorted_idx], sampled_idx[sorted_idx], sampled_dists[sorted_idx]
        
        outputs = []
        start_idx, end_idx, samples = 0, 0, 0
        while (start_idx < ray_start.size(0)):
            while (samples < self.chunk_size) and (end_idx < ray_start.size(0)):
                samples += sorted_hits[end_idx]
                end_idx += 1

            max_length = sorted_hits[end_idx - 1] + 1
            outputs += [self._forward(
                field_fn, 
                ray_start[start_idx: end_idx],
                ray_dir[start_idx: end_idx],
                (sampled_depth[start_idx: end_idx, 0: max_length], 
                 sampled_idx[start_idx: end_idx, 0: max_length],
                 sampled_dists[start_idx: end_idx, 0: max_length]),
                state
            )]
            start_idx, samples = end_idx, 0
        
        rgb, depth, missed = zip(*outputs)
        rgb, depth, missed = torch.cat(rgb, 0), torch.cat(depth, 0), torch.cat(missed)
        
        reorder_rgb, reorder_depth, reorder_missed = rgb.clone(), depth.clone(), missed.clone()  
        reorder_rgb[sorted_idx] = rgb
        reorder_depth[sorted_idx] = depth
        reorder_missed[sorted_idx] = missed
        return reorder_rgb, reorder_depth, reorder_missed

@register_model_architecture("geo_nerf", "geo_nerf")
def geo_base_architecture(args):
    args.lstm_sdf = getattr(args, "lstm_sdf", False)
    args.max_hits = getattr(args, "max_hits", 32)
    args.chunk_size = getattr(args, "chunk_size", 100)
    args.outer_chunk_size = getattr(args, "outer_chunk_size", 400 * 400)
    args.inner_chunking = getattr(args, "inner_chunking", False)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.sigmoid_activation = getattr(args, "sigmoid_activation", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)
    args.quantized_input_shuffle = getattr(args, "quantized_input_shuffle", False)
    args.expectation = getattr(args, "expectation", "rgb")
    embedding_base_architecture(args)


@register_model_architecture("geo_nerf", "geo_nerf_tri")
def geo_trilinear_architecture(args):
    args.quantized_voxel_vertex = getattr(args, "quantized_voxel_vertex", True)
    args.raypos_features = getattr(args, "raypos_features", 0)
    args.outer_chunk_size = getattr(args, "outer_chunk_size", 400 * 400)
    geo_base_architecture(args)