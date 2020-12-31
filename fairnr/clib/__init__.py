# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

''' Modified based on: https://github.com/erikwijmans/Pointnet2_PyTorch '''
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os, sys
import torch
import torch.nn.functional as F
from torch.autograd import Function
import torch.nn as nn
import sys
import numpy as np

try:
    import builtins
except:
    import __builtin__ as builtins

try:
    import fairnr.clib._ext as _ext
except ImportError:
    pass
    # raise ImportError(
    #     "Could not import _ext module.\n"
    #     "Please see the setup instructions in the README"
    # )

MAX_DEPTH = 10000.0

class BallRayIntersect(Function):
    @staticmethod
    def forward(ctx, radius, n_max, points, ray_start, ray_dir):
        inds, min_depth, max_depth = _ext.ball_intersect(
            ray_start.float(), ray_dir.float(), points.float(), radius, n_max)
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)

        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None

ball_ray_intersect = BallRayIntersect.apply


class AABBRayIntersect(Function):
    @staticmethod
    def forward(ctx, voxelsize, n_max, points, ray_start, ray_dir):
        # HACK: speed-up ray-voxel intersection by batching...
        G = min(2048, int(2 * 10 ** 9 / points.numel()))   # HACK: avoid out-of-memory
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, :H-N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, :H-N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        points = points.expand(S * G, *points.size()[1:]).contiguous()

        inds, min_depth, max_depth = _ext.aabb_intersect(
            ray_start.float(), ray_dir.float(), points.float(), voxelsize, n_max)
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)
        
        inds = inds.reshape(S, H, -1)
        min_depth = min_depth.reshape(S, H, -1)
        max_depth = max_depth.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            min_depth = min_depth[:, :N]
            max_depth = max_depth[:, :N]
        
        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None

aabb_ray_intersect = AABBRayIntersect.apply


class SparseVoxelOctreeRayIntersect(Function):
    @staticmethod
    def forward(ctx, voxelsize, n_max, points, children, ray_start, ray_dir):
        G = min(2048, int(2 * 10 ** 9 / (points.numel() + children.numel())))   # HACK: avoid out-of-memory
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, :H-N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, :H-N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        points = points.expand(S * G, *points.size()[1:]).contiguous()
        children = children.expand(S * G, *children.size()[1:]).contiguous()
        inds, min_depth, max_depth = _ext.svo_intersect(
            ray_start.float(), ray_dir.float(), points.float(), children.int(), voxelsize, n_max)
        
        min_depth = min_depth.type_as(ray_start)
        max_depth = max_depth.type_as(ray_start)
        
        inds = inds.reshape(S, H, -1)
        min_depth = min_depth.reshape(S, H, -1)
        max_depth = max_depth.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            min_depth = min_depth[:, :N]
            max_depth = max_depth[:, :N]
        
        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(min_depth)
        ctx.mark_non_differentiable(max_depth)
        return inds, min_depth, max_depth

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None

svo_ray_intersect = SparseVoxelOctreeRayIntersect.apply


class TriangleRayIntersect(Function):
    @staticmethod
    def forward(ctx, cagesize, blur_ratio, n_max, points, faces, ray_start, ray_dir):
        # HACK: speed-up ray-voxel intersection by batching...
        G = min(2048, int(2 * 10 ** 9 / (3 * faces.numel())))   # HACK: avoid out-of-memory
        S, N = ray_start.shape[:2]
        K = int(np.ceil(N / G))
        H = K * G
        if H > N:
            ray_start = torch.cat([ray_start, ray_start[:, :H-N]], 1)
            ray_dir = torch.cat([ray_dir, ray_dir[:, :H-N]], 1)
        ray_start = ray_start.reshape(S * G, K, 3)
        ray_dir = ray_dir.reshape(S * G, K, 3)
        face_points = F.embedding(faces.reshape(-1, 3), points.reshape(-1, 3))
        face_points = face_points.unsqueeze(0).expand(S * G, *face_points.size()).contiguous()
        inds, depth, uv = _ext.triangle_intersect(
            ray_start.float(), ray_dir.float(), face_points.float(), cagesize, blur_ratio, n_max)
        depth = depth.type_as(ray_start)
        uv = uv.type_as(ray_start)
        
        inds = inds.reshape(S, H, -1)
        depth = depth.reshape(S, H, -1, 3)
        uv = uv.reshape(S, H, -1)
        if H > N:
            inds = inds[:, :N]
            depth = depth[:, :N]
            uv = uv[:, :N]
        
        ctx.mark_non_differentiable(inds)
        ctx.mark_non_differentiable(depth)
        ctx.mark_non_differentiable(uv)
        return inds, depth, uv

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None

triangle_ray_intersect = TriangleRayIntersect.apply


class UniformRaySampling(Function):
    @staticmethod
    def forward(ctx, pts_idx, min_depth, max_depth, step_size, max_ray_length, deterministic=False):
        G, N, P = 256, pts_idx.size(0), pts_idx.size(1)
        H = int(np.ceil(N / G)) * G
        if H > N:
            pts_idx = torch.cat([pts_idx, pts_idx[:H-N]], 0)
            min_depth = torch.cat([min_depth, min_depth[:H-N]], 0)
            max_depth = torch.cat([max_depth, max_depth[:H-N]], 0)
        pts_idx = pts_idx.reshape(G, -1, P)
        min_depth = min_depth.reshape(G, -1, P)
        max_depth = max_depth.reshape(G, -1, P)

        # pre-generate noise
        max_steps = int(max_ray_length / step_size)
        max_steps = max_steps + min_depth.size(-1) * 2
        noise = min_depth.new_zeros(*min_depth.size()[:-1], max_steps)
        if deterministic:
            noise += 0.5
        else:
            noise = noise.uniform_()
        
        # call cuda function
        sampled_idx, sampled_depth, sampled_dists = _ext.uniform_ray_sampling(
            pts_idx, min_depth.float(), max_depth.float(), noise.float(), step_size, max_steps)
        sampled_depth = sampled_depth.type_as(min_depth)
        sampled_dists = sampled_dists.type_as(min_depth)
        
        sampled_idx = sampled_idx.reshape(H, -1)
        sampled_depth = sampled_depth.reshape(H, -1)
        sampled_dists = sampled_dists.reshape(H, -1)
        if H > N:
            sampled_idx = sampled_idx[: N]
            sampled_depth = sampled_depth[: N]
            sampled_dists = sampled_dists[: N]
        
        max_len = sampled_idx.ne(-1).sum(-1).max()
        sampled_idx = sampled_idx[:, :max_len]
        sampled_depth = sampled_depth[:, :max_len]
        sampled_dists = sampled_dists[:, :max_len]

        ctx.mark_non_differentiable(sampled_idx)
        ctx.mark_non_differentiable(sampled_depth)
        ctx.mark_non_differentiable(sampled_dists)
        return sampled_idx, sampled_depth, sampled_dists

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None

uniform_ray_sampling = UniformRaySampling.apply


class InverseCDFRaySampling(Function):
    @staticmethod
    def forward(ctx, pts_idx, min_depth, max_depth, probs, steps, fixed_step_size=-1, deterministic=False):
        G, N, P = 200, pts_idx.size(0), pts_idx.size(1)
        H = int(np.ceil(N / G)) * G
        
        if H > N:
            pts_idx = torch.cat([pts_idx, pts_idx[:1].expand(H-N, P)], 0)
            min_depth = torch.cat([min_depth, min_depth[:1].expand(H-N, P)], 0)
            max_depth = torch.cat([max_depth, max_depth[:1].expand(H-N, P)], 0)
            probs = torch.cat([probs, probs[:1].expand(H-N, P)], 0)
            steps = torch.cat([steps, steps[:1].expand(H-N)], 0)
        # print(G, P, np.ceil(N / G), N, H, pts_idx.shape, min_depth.device)
        pts_idx = pts_idx.reshape(G, -1, P)
        min_depth = min_depth.reshape(G, -1, P)
        max_depth = max_depth.reshape(G, -1, P)
        probs = probs.reshape(G, -1, P)
        steps = steps.reshape(G, -1)

        # pre-generate noise
        max_steps = steps.ceil().long().max() + P
        noise = min_depth.new_zeros(*min_depth.size()[:-1], max_steps)
        if deterministic:
            noise += 0.5
        else:
            noise = noise.uniform_().clamp(min=0.001, max=0.999)  # in case
        
        # call cuda function
        chunk_size = 4 * G  # to avoid oom?
        results = [
            _ext.inverse_cdf_sampling(
                pts_idx[:, i:i+chunk_size].contiguous(), 
                min_depth.float()[:, i:i+chunk_size].contiguous(), 
                max_depth.float()[:, i:i+chunk_size].contiguous(), 
                noise.float()[:, i:i+chunk_size].contiguous(), 
                probs.float()[:, i:i+chunk_size].contiguous(), 
                steps.float()[:, i:i+chunk_size].contiguous(), 
                fixed_step_size)
            for i in range(0, min_depth.size(1), chunk_size)
        ]
        sampled_idx, sampled_depth, sampled_dists = [
            torch.cat([r[i] for r in results], 1)
            for i in range(3)
        ]
        sampled_depth = sampled_depth.type_as(min_depth)
        sampled_dists = sampled_dists.type_as(min_depth)
        
        sampled_idx = sampled_idx.reshape(H, -1)
        sampled_depth = sampled_depth.reshape(H, -1)
        sampled_dists = sampled_dists.reshape(H, -1)
        if H > N:
            sampled_idx = sampled_idx[: N]
            sampled_depth = sampled_depth[: N]
            sampled_dists = sampled_dists[: N]
        
        max_len = sampled_idx.ne(-1).sum(-1).max()
        sampled_idx = sampled_idx[:, :max_len]
        sampled_depth = sampled_depth[:, :max_len]
        sampled_dists = sampled_dists[:, :max_len]

        ctx.mark_non_differentiable(sampled_idx)
        ctx.mark_non_differentiable(sampled_depth)
        ctx.mark_non_differentiable(sampled_dists)
        return sampled_idx, sampled_depth, sampled_dists

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None, None, None, None, None

inverse_cdf_sampling = InverseCDFRaySampling.apply


# back-up for ray point sampling
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

