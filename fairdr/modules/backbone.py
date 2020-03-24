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
from fairdr.data.data_utils import load_matrix
from fairdr.modules.pointnet2.pointnet2_utils import furthest_point_sample
from fairdr.modules.linear import Linear, Embedding, PosEmbLinear
from fairseq import options, utils
from fairseq.modules import (
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
    LayerNorm
)


class Backbone(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, pointcloud, add_dummy=False):
        feats, xyz = self._forward(pointcloud)
        return feats, xyz

    def _forward(self, pointcloud):
        raise NotImplementedError


class QuantizedEmbeddingBackbone(Backbone):
    """
    Embeddings on fixed voxel models
    """
    def __init__(self, args):
        super().__init__(args)
        self.embed_dim = args.quantized_embed_dim
        self.quantized_input_shuffle = getattr(args, "quantized_input_shuffle", True)
        self.sample_points = getattr(args, 'quantized_subsampling_points', None)
        self.voxel_path = args.quantized_voxel_path if args.quantized_voxel_path is not None \
            else os.path.join(args.data, 'voxel.txt')
        assert os.path.exists(self.voxel_path), "voxel file does not exist"
        
        self.keys = nn.Parameter(torch.from_numpy(load_matrix(self.voxel_path)[:, 3:]), requires_grad=False)
        self.values = Embedding(self.keys.size(0), self.embed_dim, None)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-subsampling-points', type=int, metavar='N',
                            help='if not set (None), do not perform subsampling.')
        parser.add_argument('--quantized-voxel-path', type=str,
                            help="path to a pre-computed voxels.")
        parser.add_argument('--quantized-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--quantized-input-shuffle', action='store_true')

    def _forward(self, pointcloud: torch.cuda.FloatTensor, add_dummy=False):
        if self.sample_points > 0:
            assert pointcloud.size(1) >= self.sample_points, "need more points"
            if self.training and self.quantized_input_shuffle:
                rand_inds = pointcloud.new_zeros(*pointcloud.size()[:-1]).uniform_().sort(1)[1]
                pointcloud = pointcloud.gather(1, rand_inds.unsqueeze(2).expand_as(pointcloud))
            pointcloud = pointcloud.gather(
                1, furthest_point_sample(
                    pointcloud, self.sample_points).unsqueeze(2).expand(
                        pointcloud.size(0), self.sample_points, 3).long())
        _, ids = ((pointcloud[:, :, None, :] - self.keys[None, None, :, :]) ** 2).sum(-1).min(-1)
        return self.values(ids), pointcloud   

    @property
    def feature_dim(self):
        return self.embed_dim

class Pointnet2Backbone(Backbone):
    """
    backbone network for pointcloud feature learning.
    this is a Pointnet++ single-scale grouping network
    --- copying from somewhere, maybe not optimal at all...
    """
    def __init__(self, args):
        super().__init__(args)
        self.r =getattr(args, "pointnet2_min_radius", 0.1)
        self.input_feature_dim = getattr(args, "pointnet2_input_feature_dim", 0)
        self.input_shuffle = getattr(args, "pointnet2_input_shuffle", False)
        self.upsample512 = getattr(args, "pointnet2_upsample512", False)
        self.sa1 = PointnetSAModuleVotes(
                npoint=512,
                radius=self.r,
                nsample=64,
                mlp=[self.input_feature_dim, 32, 32, 64],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=256,
                radius=self.r*2,
                nsample=32,
                mlp=[64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=64,
                radius=self.r*4,
                nsample=16,
                mlp=[128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=16,
                radius=self.r*8,
                nsample=16,
                mlp=[256, 256, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,128])
        self.fp2 = PointnetFPModule(mlp=[128+128,128,64])

        if self.upsample512:
            self.fp3 = PointnetFPModule(mlp=[64+64,128,64])

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    @staticmethod
    def add_args(parser):
        parser.add_argument('--pointnet2-min-radius', type=float, metavar='D')
        parser.add_argument('--pointnet2-input-feature-dim', type=int, metavar='N')
        parser.add_argument('--pointnet2-input-shuffle', action='store_true')
        parser.add_argument('--pointnet2-upsample512', action='store_true')

    def _forward(self, pointcloud: torch.cuda.FloatTensor, add_dummy=False):
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
        end_points = {}
        batch_size = pointcloud.shape[0]
        if self.input_shuffle and self.training:
            rand_inds = pointcloud.new_zeros(*pointcloud.size()[:-1]).uniform_().sort(1)[1]
            pointcloud = pointcloud.gather(1, rand_inds.unsqueeze(2).expand_as(pointcloud))

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

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

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        xyz = end_points['sa2_xyz']
        if self.upsample512:
            features = self.fp3(end_points['sa1_xyz'], end_points['sa2_xyz'], end_points['sa1_features'], features)
            xyz = end_points['sa1_xyz']
        feats = features.transpose(1, 2)
        return feats, xyz

    @property
    def feature_dim(self):
        return 64


class TransformerBackbone(Backbone):
    
    def __init__(self, args):
        super().__init__(args)
        self.dropout = args.dropout
        self.sample_points = getattr(args, 'subsampling_points', None)
        self.furthest_sampling = getattr(args, 'furthest_sampling', True)
        self.input_shuffle = getattr(args, 'transformer_input_shuffle', False)
        self.transformer_pos_embed = getattr(args, 'transformer_pos_embed', False)
        self.layers = nn.ModuleList([])
        self.layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        
        embed_dim = self.args.encoder_embed_dim
        if not self.transformer_pos_embed:
            self.point_embed = Linear(3, embed_dim)
        else:
            self.point_embed = PosEmbLinear(3, embed_dim)
        
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--subsampling-points', type=int, metavar='N',
                            help='if not set (None), do not perform subsampling.')
        parser.add_argument('--furthest-sampling', action='store_true', 
                            help='if enabled, use furthest sampling instead of random sampling')
        parser.add_argument('--transformer-input-shuffle', action='store_true',
                            help='if use subsampling, do we need to shuffle the points?')
        parser.add_argument("--transformer-pos-embed", action='store_true', 
                            help='use positional embedding instead of linear projection')

        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        
    def _forward(self, pointcloud: torch.cuda.FloatTensor, add_dummy=False):
        if self.sample_points > 0:
            assert pointcloud.size(1) >= self.sample_points, "need more points"
            if self.training and self.input_shuffle:
                rand_inds = pointcloud.new_zeros(*pointcloud.size()[:-1]).uniform_().sort(1)[1]
                pointcloud = pointcloud.gather(1, rand_inds.unsqueeze(2).expand_as(pointcloud))
            
            if self.furthest_sampling:
                pointcloud = pointcloud.gather(
                    1, furthest_point_sample(
                        pointcloud, self.sample_points).unsqueeze(2).expand(
                            pointcloud.size(0), self.sample_points, 3).long())
            else:
                assert self.input_shuffle, "only supports input shuffle"
                pointcloud = pointcloud[:, :self.sample_points]
        
        # transform: B x N x 3 --> B x N x D
        x = self.point_embed(pointcloud)
        x = x.transpose(0, 1)
    
        for layer in self.layers:
            x = layer(x, None)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        
        x = x.transpose(0, 1)  
        return x, pointcloud    # return features & pointcloud

    @property
    def feature_dim(self):
        return self.args.encoder_embed_dim