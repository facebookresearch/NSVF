# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import random

from fairnr.data.geometry import get_ray_direction

TINY = 1e-9


class Reader(nn.Module):
    """
    basic image reader
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_pixels = args.pixel_per_view

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pixel-per-view', type=float, metavar='N', 
                            help='number of pixels sampled for each view')
        parser.add_argument("--sampling-on-mask", nargs='?', const=0.9, type=float,
                            help="this value determined the probability of sampling rays on masks")
        parser.add_argument("--sampling-at-center", type=float,
                            help="only useful for training where we restrict sampling at center of the image")
        parser.add_argument("--sampling-on-bbox", action='store_true',
                            help="sampling points to close to the mask")
        parser.add_argument("--sampling-patch-size", type=int, 
                            help="sample pixels based on patches instead of independent pixels")
        parser.add_argument("--sampling-skipping-size", type=int,
                            help="sample pixels if we have skipped pixels")

    def forward(self, uv, intrinsics, extrinsics, **kwargs):
        uv = self.sample_pixels(uv, **kwargs)  # S x V x 2 x (N x P x P)
        S, V = uv.size()[:2]
        flatten_uv = uv.reshape(S, V, 2, -1)
        ray_start = extrinsics[:, :, :3, 3]
        ray_dir = torch.stack([torch.stack([
                get_ray_direction(ray_start[s, v], flatten_uv[s, v], intrinsics[s], extrinsics[s, v], 1)
            for v in range(V)]) for s in range(S)])
        return ray_start.unsqueeze(-2), ray_dir.transpose(2, 3), uv
    
    @torch.no_grad()
    def sample_pixels(self, uv, alpha, size, mask=None, **kwargs):
        H, W = int(size[0,0,0]), int(size[0,0,1])
        S, V = alpha.size()[:2]
        if not self.training:
            return uv.reshape(S, V, 2, 1, H, W)

        if mask is None:
            mask = (alpha > 0)
        mask = mask.float().reshape(S, V, H, W)

        if self.args.sampling_at_center < 1.0:
            r = (1 - self.args.sampling_at_center) / 2.0
            mask0 = mask.new_zeros(S, V, H, W)
            mask0[:, :, int(H * r): H - int(H * r), int(W * r): W - int(W * r)] = 1
            mask = mask * mask0
        
        if self.args.sampling_on_bbox:
            x_has_points = mask.sum(2, keepdim=True) > 0
            y_has_points = mask.sum(3, keepdim=True) > 0
            mask = (x_has_points & y_has_points).float()  

        probs = mask / (mask.sum() + 1e-8)
        if self.args.sampling_on_mask > 0.0:
            probs = self.args.sampling_on_mask * probs + (1 - self.args.sampling_on_mask) * 1.0 / (H * W)

        num_pixels = int(self.args.pixel_per_view)
        patch_size, skip_size = self.args.sampling_patch_size, self.args.sampling_skipping_size
        C = patch_size * skip_size
        
        if C > 1:
            probs = probs.reshape(S, V, H // C, C, W // C, C).sum(3).sum(-1)
            num_pixels = num_pixels // patch_size // patch_size

        flatten_probs = probs.reshape(S, V, -1) 
        sampled_index = sampling_without_replacement(torch.log(flatten_probs+ TINY), num_pixels)
        sampled_masks = torch.zeros_like(flatten_probs).scatter_(-1, sampled_index, 1).reshape(S, V, H // C, W // C)

        if C > 1:
            sampled_masks = sampled_masks[:, :, :, None, :, None].repeat(
                1, 1, 1, patch_size, 1, patch_size).reshape(S, V, H // skip_size, W // skip_size)
            if skip_size > 1:
                full_datamask = sampled_masks.new_zeros(S, V, skip_size * skip_size, H // skip_size, W // skip_size)
                full_index = torch.randint(skip_size*skip_size, (S, V))
                for i in range(S):
                    for j in range(V):
                        full_datamask[i, j, full_index[i, j]] = sampled_masks[i, j]
                sampled_masks = full_datamask.reshape(
                    S, V, skip_size, skip_size, H // skip_size, W // skip_size).permute(0, 1, 4, 2, 5, 3).reshape(S, V, H, W)
        
        X, Y = uv[:,:,0].reshape(S, V, H, W), uv[:,:,1].reshape(S, V, H, W)
        X = X[sampled_masks>0].reshape(S, V, 1, -1, patch_size, patch_size)
        Y = Y[sampled_masks>0].reshape(S, V, 1, -1, patch_size, patch_size)
        return torch.cat([X, Y], 2)


def sampling_without_replacement(logp, k):
    def gumbel_like(u):
        return -torch.log(-torch.log(torch.rand_like(u) + TINY) + TINY)
    scores = logp + gumbel_like(logp)
    return scores.topk(k, dim=-1)[1]