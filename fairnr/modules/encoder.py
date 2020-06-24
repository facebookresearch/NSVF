# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
import math
import logging
logger = logging.getLogger(__name__)


from fairnr.data.data_utils import load_matrix, unique_points
from fairnr.data.geometry import (
    trilinear_interp, splitting_points, offset_points
)
from fairnr.modules.pointnet2.pointnet2_utils import aabb_ray_intersect
from fairnr.modules.linear import FCBlock, Linear, Embedding

MAX_DEPTH = 10000.0


class Encoder(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_context = getattr(args, "context", None)
        self.online_pruning = getattr(args, "online_pruning", False)

    def forward(self, **kwargs):
        raise NotImplementedError
    

class SparseVoxelEncoder(Encoder):

    def __init__(self, args):
        super().__init__(args)

        self.voxel_path = args.voxel_path if args.voxel_path is not None \
            else os.path.join(args.data, 'voxel.txt')
        assert os.path.exists(self.voxel_path), "Initial voxel file does not exist..."

        half_voxel = args.voxel_size * .5
        fine_points = torch.from_numpy(load_matrix(self.voxel_path)[:, 3:])
        fine_length = fine_points.size(0)
 
        # transform from voxel centers to voxel corners (key/values)
        fine_coords = (fine_points / half_voxel).floor_().long()
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats  = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)
        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()
        
        # set total size  # maximum number of voxel allowed
        self.total_size = getattr(args, "total_num_embedding", None)
        if self.total_size is None:
            self.total_size = fine_keys.size(0)

        points, feats = torch.zeros(self.total_size, 3), torch.zeros(self.total_size, 8).long()
        keys, keep = torch.zeros(self.total_size, 3).long(), torch.zeros(self.total_size).long()    

        # assign values
        points[: fine_length] = fine_points
        feats[: fine_length] = fine_feats
        keep[: fine_length] = 1
        keys[: num_keys] = fine_keys

        # set-up hyperparameters
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.values = Embedding(self.total_size, self.embed_dim, None)
        self.deterministic_step = getattr(args, "deterministic_step", False)

        # register parameters
        self.register_buffer("points", points)   # voxel centers
        self.register_buffer("feats", feats)     # for each voxel, 8 vertexs
        self.register_buffer("keys", keys)
        self.register_buffer("keep", keep)
        self.register_buffer("num_keys", num_keys)

        self.register_buffer("voxel_size", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(args.raymarching_stepsize))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

    def upgrade_state_dict_named(self, state_dict, name):
        # update the voxel embedding shapes
        loaded_values = state_dict[name + '.values.weight']
        self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
        self.values.num_embeddings = self.values.weight.size(0)
        self.total_size = self.values.weight.size(0)
        self.num_keys = self.num_keys * 0 + self.total_size
        
        # update the buffers shapes
        self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
        self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
        self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
        self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())

    @staticmethod
    def add_args(parser):
        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial')
        parser.add_argument('--voxel-path', type=str, help='path for initial voxel file')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--total-num-embedding', type=int, metavar='N', help='totoal number of embeddings to initialize')
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='N', help='ray marching step size for sparse voxels')

    def precompute(self, id=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        keys   = self.keys[: self.num_keys]
       
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous()

            # moving to multiple objects
            feats = feats + values.size(1) * torch.arange(values.size(0), 
                device=feats.device, dtype=feats.dtype)[:, None, None]
        return feats, points, values

    def ray_voxel_intersect(self, ray_start, ray_dir, point_xyz, point_feats):
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3)
        ray_dir = ray_dir.reshape(S, V * P, 3)
        pts_idx, min_depth, max_depth = aabb_ray_intersect(
            self.voxel_size, self.max_hits, point_xyz, ray_start, ray_dir)
       
        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        # extend the point-index to multiple shapes
        pts_idx = (pts_idx + H * torch.arange(S, 
            device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
            ).masked_fill_(pts_idx.eq(-1), -1)
        return ray_start, ray_dir, min_depth, max_depth, pts_idx, hits

    def ray_sample(self, pts_idx, min_depth, max_depth):
        # ray_sampler = uniform_ray_sampling
        sampled_idx, sampled_depth, sampled_dists = parallel_ray_sampling(
            self.step_size, pts_idx, min_depth, max_depth, 
            self.deterministic_step or (not self.training))
        sampled_dists = sampled_dists.clamp(min=0.0)
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        return sampled_depth, sampled_idx, sampled_dists

    def forward(self, samples, encoder_states):
        point_feats, point_xyz, values = encoder_states
        sampled_xyz, sampled_idx = samples
        sampled_idx = sampled_idx.long()
        
        # resample point features
        point_xyz = point_xyz.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), 3))
        point_feats = point_feats.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), point_feats.size(-1)))
        point_feats = F.embedding(point_feats, values).view(point_feats.size(0), -1)

        # tri-linear interpolation
        p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
        q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5   # BUG? 
        return trilinear_interp(p, q, point_feats)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5):
        logger.info("pruning...")
        feats, points, values = self.precompute(id=None)
        chunk_size, bits = 64, 16

        def prune_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_idx = torch.arange(points.size(0), device=points.device)[:, None].expand(*sampled_xyz.size()[:2])
            sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)
            sampled_feats = self.forward((sampled_xyz, sampled_idx), (feats, points, values))  # get field inputs
            
            # evaluation with density
            sigma = field_fn(sampled_feats, outputs=['sigma'])[0]
            free_energy = -torch.relu(sigma).reshape(-1, bits ** 3).max(-1)[0]
            
            # prune voxels if needed
            return (1 - torch.exp(free_energy)) > th

        keep = torch.cat([prune_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)
        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        logger.info("pruning done. # of voxels before: {}, after: {} voxels".format(points.size(0), keep.sum()))

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        feats, points, values = self.precompute(id=None)

        new_points, new_feats, new_values = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_values.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        self.values.weight = nn.Parameter(new_values)
        self.values.num_embeddings = self.values.weight.size(0)
        self.total_size = self.values.weight.size(0)
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)
        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(points.size(0), self.keep.sum()))
        
    @property
    def feature_dim(self):
        return self.embed_dim

    @property
    def dummy_loss(self):
        return self.values.weight[0,0] * 0.0


class MultiSparseVoxelEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)
        self.voxel_lists = open(args.quantized_voxel_path).readlines()
        self.backbones = nn.ModuleList(
            [SparseVoxelEncoder(args, vox.strip()) for vox in self.voxel_lists])
        self.current_id = None
        self.offset = self.backbones[0].offset

    def forward(self, id, *args, **kwargs):
        assert id.size(0) == 1, "for now, only works for one object"
        self.current_id = id[0]
        return self.backbones[id[0]].forward(id, *args, **kwargs)
    
    def get_features(self, x, values):
        return F.embedding(x, values)

    def pruning(self, keep):
        self.backbones[self.current_id].pruning(keep)

    def splitting(self, *args, **kwargs):
        for i in range(len(self.backbones)):
            self.backbones[i].splitting(*args, **kwargs)

    @property
    def feature_dim(self):
        return self.backbones[0].embed_dim

    @property
    def keep(self):
        return torch.cat([b.keep for b in self.backbones], -1)

    @property
    def dummy_loss(self):
        return sum([b.dummy_loss for b in self.backbones])

    @staticmethod
    def add_args(parser):
        pass



@torch.no_grad()
def _parallel_ray_sampling(MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=False):
    # uniform sampling
    _min_depth = min_depth.min(1)[0]
    _max_depth = max_depth.masked_fill(max_depth.eq(MAX_DEPTH), 0).max(1)[0]
    max_ray_length = (_max_depth - _min_depth).max()
    
    delta = torch.arange(int(max_ray_length / MARCH_SIZE), device=min_depth.device, dtype=min_depth.dtype)
    delta = delta[None, :].expand(min_depth.size(0), delta.size(-1))
    if deterministic:
        delta = delta + 0.5
    else:
        delta = delta + delta.clone().uniform_().clamp(min=0.01, max=0.99)
    delta = delta * MARCH_SIZE
    sampled_depth = min_depth[:, :1] + delta
    sampled_idx = (sampled_depth[:, :, None] >= min_depth[:, None, :]).sum(-1) - 1
    sampled_idx = pts_idx.gather(1, sampled_idx)    
    
    # include all boundary points
    sampled_depth = torch.cat([min_depth, max_depth, sampled_depth], -1)
    sampled_idx = torch.cat([pts_idx, pts_idx, sampled_idx], -1)

    # reorder
    sampled_depth, ordered_index = sampled_depth.sort(-1)
    sampled_idx = sampled_idx.gather(1, ordered_index)
    sampled_dists = sampled_depth[:, 1:] - sampled_depth[:, :-1]          # distances
    sampled_depth = .5 * (sampled_depth[:, 1:] + sampled_depth[:, :-1])   # mid-points

    # remove all invalid depths
    min_ids = (sampled_depth[:, :, None] >= min_depth[:, None, :]).sum(-1) - 1
    max_ids = (sampled_depth[:, :, None] >= max_depth[:, None, :]).sum(-1)

    sampled_depth.masked_fill_(
        (max_ids.ne(min_ids)) |
        (sampled_depth > _max_depth[:, None]) |
        (sampled_dists == 0.0)
        , MAX_DEPTH)
    sampled_depth, ordered_index = sampled_depth.sort(-1) # sort again
    sampled_masks = sampled_depth.eq(MAX_DEPTH)
    num_max_steps = (~sampled_masks).sum(-1).max()
    
    sampled_depth = sampled_depth[:, :num_max_steps]
    sampled_dists = sampled_dists.gather(1, ordered_index).masked_fill_(sampled_masks, 0.0)[:, :num_max_steps]
    sampled_idx = sampled_idx.gather(1, ordered_index).masked_fill_(sampled_masks, -1)[:, :num_max_steps]
    
    return sampled_idx, sampled_depth, sampled_dists


@torch.no_grad()
def parallel_ray_sampling(MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=False):
    chunk_size=4096
    full_size = min_depth.shape[0]
    if full_size <= chunk_size:
        return _parallel_ray_sampling(MARCH_SIZE, pts_idx, min_depth, max_depth, deterministic=deterministic)

    outputs = zip(*[
            _parallel_ray_sampling(
                MARCH_SIZE, 
                pts_idx[i:i+chunk_size], min_depth[i:i+chunk_size], max_depth[i:i+chunk_size],
                deterministic=deterministic) 
            for i in range(0, full_size, chunk_size)])
    sampled_idx, sampled_depth, sampled_dists = outputs
    
    def padding_points(xs, pad):
        if len(xs) == 1:
            return xs[0]
        
        maxlen = max([x.size(1) for x in xs])
        full_size = sum([x.size(0) for x in xs])
        xt = xs[0].new_ones(full_size, maxlen).fill_(pad)
        st = 0
        for i in range(len(xs)):
            xt[st: st + xs[i].size(0), :xs[i].size(1)] = xs[i]
            st += xs[i].size(0)
        return xt

    sampled_idx = padding_points(sampled_idx, -1)
    sampled_depth = padding_points(sampled_depth, MAX_DEPTH)
    sampled_dists = padding_points(sampled_dists, 0.0)
    return sampled_idx, sampled_depth, sampled_dists
