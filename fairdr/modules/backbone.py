# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

from fairdr.modules.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes, PointnetFPModule
)


class Pointnet2Backbone(nn.Module):
    """
    backbone network for pointcloud feature learning.
    this is a Pointnet++ single-scale grouping network
    """
    def __init__(self, input_feature_dim=0, min_radius=0.1):
        super().__init__()

        self.r = min_radius
        self.sa1 = PointnetSAModuleVotes(
                npoint=512,
                radius=r,
                nsample=64,
                mlp=[input_feature_dim, 32, 32, 64],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=256,
                radius=r*2,
                nsample=32,
                mlp=[64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=64,
                radius=r*4,
                nsample=16,
                mlp=[128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=16,
                radius=r*8,
                nsample=16,
                mlp=[256, 256, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,128])
        self.fp2 = PointnetFPModule(mlp=[128+128,128,64])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )

        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points: end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        from fairseq import pdb; pdb.set_trace()

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        end_points['fp2_features'] = features
        end_points['fp2_xyz'] = end_points['sa2_xyz']
        num_seed = end_points['fp2_xyz'].shape[1]
        end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds
        return end_points


