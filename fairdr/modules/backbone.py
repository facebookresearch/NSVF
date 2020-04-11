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
from fairdr.data.data_utils import load_matrix, unique_points
from fairdr.data.geometry import trilinear_interp
from fairdr.modules.pointnet2.pointnet2_utils import furthest_point_sample
from fairdr.modules.linear import Linear, Embedding, PosEmbLinear, NeRFPosEmbLinear
from fairseq import options, utils
from fairseq.modules import (
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer,
    LayerNorm
)


BACKBONE_REGISTRY = {}


def register_backnone(name):
    def register_backbone_cls(cls):
        if name in BACKBONE_REGISTRY:
            raise ValueError('Cannot register duplicate backbone module ({})'.format(name))
        BACKBONE_REGISTRY[name] = cls
        return cls
    return register_backbone_cls

@register_backnone('base_backbone')
class Backbone(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, *args, **kwargs):
        feats, xyz, values = self._forward(*args, **kwargs)

        # placeholder reserved for backbone independent functions
        return feats, xyz, values

    def _forward(self, pointcloud):
        raise NotImplementedError
    
    def get_features(self, x):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-voxel-path', type=str, help="path to a pre-computed voxels.")
    
    @torch.no_grad()
    def pruning(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def splitting(self, *args, **kwargs):
        raise NotImplementedError


@register_backnone('dynamic_embedding')
class DynamicEmbeddingBackbone(Backbone):

    def __init__(self, args):
        super().__init__(args)

        self.voxel_path = args.quantized_voxel_path if args.quantized_voxel_path is not None \
            else os.path.join(args.data, 'voxel.txt')
        assert os.path.exists(self.voxel_path), "Initial voxel file does not exist..."

        self.voxel_size = args.voxel_size
        self.total_size = 12000   # maximum number of voxel allowed
        self.half_voxel = self.voxel_size * .5

        points, feats = torch.zeros(self.total_size, 3), torch.zeros(self.total_size, 8).long()
        keys, keep = torch.zeros(self.total_size, 3).long(), torch.zeros(self.total_size).long()

        init_points = torch.from_numpy(load_matrix(self.voxel_path)[:, 3:])
        init_length = init_points.size(0)

        # working in LONG space
        init_coords = (init_points / self.half_voxel).floor_().long()
        offset = torch.tensor([[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [-1., 1., 1.],
                            [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.], [-1., -1., -1.]]).long()
        init_keys0 = (init_coords.unsqueeze(1) + offset.unsqueeze(0)).reshape(-1, 3)
        init_keys  = unique_points(init_keys0)
        init_feats = (init_keys0[:, None, :] - init_keys[None, :, :]).float().norm(dim=-1).min(1)[1].reshape(-1, 8)
        
        points[: init_length] = init_points
        feats[: init_length] = init_feats
        keep[: init_length] = 1
        keys[: init_keys.size(0)] = init_keys

        self.register_buffer("points", points)   # voxel centers
        self.register_buffer("feats", feats)     # for each voxel, 8 vertexs
        self.register_buffer("keys", keys)
        self.register_buffer("keep", keep)
        self.register_buffer("offset", offset)
        self.register_buffer("num_keys", torch.scalar_tensor(init_keys.size(0)).long())

        # voxel embeddings
        self.embed_dim = getattr(args, "quantized_embed_dim", None)
        if self.embed_dim is not None:
            self.values = Embedding(self.total_size, self.embed_dim, None)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-embed-dim', type=int, metavar='N', help="embedding size")

    def _forward(self, *args, **kwargs):
        return self.feats[self.keep.bool()].unsqueeze(0), \
               self.points[self.keep.bool()].unsqueeze(0), \
               self.values.weight if self.values is not None else None

    def get_features(self, x, values):
        return F.embedding(x, values)

    @torch.no_grad()
    def pruning(self, keep):
        self.keep.masked_scatter_(self.keep.bool(), keep.long())

    @torch.no_grad()
    def splitting(self, half_voxel, update_buffer=True):
        offset = self.offset
        point_xyz = self.points[self.keep.bool()]
        point_feats = self.values(self.feats[self.keep.bool()])
       
        # generate new centers
        half_voxel = half_voxel * .5
        new_points = (point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * half_voxel).reshape(-1, 3)
        
        old_coords = (point_xyz / half_voxel).floor_().long()
        new_coords = (old_coords.unsqueeze(1) + offset.unsqueeze(0)).reshape(-1, 3)
        
        new_keys0 = (new_coords.unsqueeze(1) + offset.unsqueeze(0)).reshape(-1, 3)
        new_keys = unique_points(new_keys0)
        new_feats = (new_keys0[:, None, :] - new_keys[None, :, :]).float().norm(dim=-1).min(1)[1].reshape(-1, 8)
        
        # recompute key vectors using trilinear interpolation 
        new_keys_idx = (new_keys[:, None, :] - old_coords[None, :, :]).float().norm(dim=-1).min(1)[1]
        p = (new_keys - old_coords[new_keys_idx]).type_as(point_xyz).unsqueeze(1) * .25 + 0.5 # (1/4 voxel size)
        q = (self.offset.type_as(p) * .5 + .5).unsqueeze(0)   # (1/2 voxel size)
        new_values = trilinear_interp(p, q, point_feats[new_keys_idx])
        
        # assign to the parameters
        # TODO: safe assignment. do not over write the original embeddings
        new_num_keys = new_values.size(0)
        new_point_length = new_points.size(0)

        if update_buffer:
            self.values.weight.data.index_copy_(
                0,
                torch.arange(new_num_keys, device=new_values.device),
                new_values.data
            )

            self.points[: new_point_length] = new_points
            self.feats[: new_point_length] = new_feats
            self.keep = torch.zeros_like(self.keep)
            self.keep[: new_point_length] = 1
            self.num_keys += (new_num_keys - self.num_keys)

    @property
    def feature_dim(self):
        return self.embed_dim


@register_backnone("transformer")
class TransformerBackbone(DynamicEmbeddingBackbone):
    
    def __init__(self, args):
        super().__init__(args)

        # additional transformer parameters
        self.dropout = args.dropout
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.embed_dim = self.args.encoder_embed_dim
        self.point_embed = PosEmbLinear(3, self.embed_dim, no_linear=False, scale=10.0)
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

        # remove voxel embeddings
        self.values = None

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

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
        
    def _forward(self, *args, **kwargs):
        feats, points, _ = super()._forward()

        # get inputs
        keys = self.keys[: self.num_keys].clone().type_as(points).unsqueeze(0).requires_grad_()

        # embed points ---> features
        x = self.point_embed(keys)
        x = x.transpose(0, 1)
        
        padding_mask = x.new_zeros(x.size(1), x.size(0)).bool()
        for layer in self.layers:
            x = layer(x, padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)  
        
        values = x[0]
        return feats, points, values

    @property
    def feature_dim(self):
        return self.args.encoder_embed_dim