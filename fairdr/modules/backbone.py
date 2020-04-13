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
from fairdr.modules.linear import FCLayer, Linear, Embedding, PosEmbLinear, NeRFPosEmbLinear
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
        self.use_context = getattr(args, "context", None)
        self.online_pruning = getattr(args, "online_pruning", False)

    def forward(self, *args, **kwargs):
        # placeholder reserved for backbone independent functions
        return self._forward(*args, **kwargs)

    def _forward(self, pointcloud):
        raise NotImplementedError
    
    def get_features(self, x):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-voxel-path', type=str, help="path to a pre-computed voxels.")
        parser.add_argument('--context', choices=['id', 'image'], help='context for backbone model.')
        parser.add_argument('--online-pruning', action='store_true', help='if set, model performs online pruning')

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
        init_keys, init_feats  = torch.unique(init_keys0, dim=0, sorted=True, return_inverse=True)
        init_feats = init_feats.reshape(-1, 8)
        
        points[: init_length] = init_points
        feats[: init_length] = init_feats
        keep[: init_length] = 1
        keys[: init_keys.size(0)] = init_keys

        self.register_buffer("points", points)   # voxel centers
        self.register_buffer("feats", feats)     # for each voxel, 8 vertexs
        self.register_buffer("keys", keys)
        self.register_buffer("keep", keep)
        self.register_buffer("offset", offset)
        self.register_buffer("num_keys", 
            torch.scalar_tensor(init_keys.size(0)).long())

        # voxel embeddings
        self.embed_dim = getattr(args, "quantized_embed_dim", None)
        if self.embed_dim is not None:
            self.values = Embedding(self.total_size, self.embed_dim, None)

        if self.use_context is not None and self.use_context == 'id':
            assert self.args.total_num_context > 0, "index embeddings for different frames"
            self.context_embed = Embedding(self.args.total_num_context, self.embed_dim, None)

        if self.online_pruning:
            self.pproj_dim = getattr(args, "quantized_pproj_dim", 64)
            self.p1 = FCLayer(self.embed_dim, self.pproj_dim)
            self.p2 = Linear(self.pproj_dim * 8, 1)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--quantized-pproj-dim', type=int, metavar='N', help="only useful if online_pruning set")
        parser.add_argument('--total-num-context', type=int, metavar='N', help="if use id as inputs, unique embeddings")

    def _forward(self, id, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[: self.num_keys] if self.values is not None else None

        # extend size to support multi-objects
        feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
        points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
        values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous()
        
        # object dependent value
        if self.use_context is not None:
            if self.use_context == 'id':
                values = values + self.context_embed(id).unsqueeze(1)
            else:
                raise NotImplementedError("only add-on condition for now.")

        feats = feats + values.size(1) * torch.arange(values.size(0), 
            device=feats.device, dtype=feats.dtype)[:, None, None]
        values = values.view(-1, values.size(-1))  # reshape to 2D

        if not self.online_pruning:
            return feats, points, values, None

        # predict keep or remove using central features
        centers = self.get_features(feats, values)  # S x P x 8 x D
        predicts = self.p2(self.p1(centers).view(*centers.size()[:2], -1)).squeeze(-1)

        # feats = feats[predicts > 0].unsqueeze(0)
        # points = points[predicts > 0].unsqueeze(0)
        # predicts = predicts[predicts > 0].unsqueeze(0)
        # from fairseq import pdb; pdb.set_trace()

        # online split
        # new_points, new_feats, new_values = self.splitting(self.voxel_size * .5, False, (points[0], feats[0], values[0]))
        # from fairseq import pdb; pdb.set_trace()        
        return feats, points, values, predicts
              
    def get_features(self, x, values):
        return F.embedding(x, values)

    def pruning(self, keep):
        self.keep.masked_scatter_(self.keep.bool(), keep.long())

    def splitting(self, half_voxel, update_buffer=True, features=None):
        if features is not None:
            point_xyz, feats, values = features
            point_feats = F.embedding(feats, values)
        else:
            point_xyz, feats = self.points[self.keep.bool()], self.feats[self.keep.bool()]
            point_feats = self.values(feats)

        offset = self.offset
        
        # generate new centers
        half_voxel = half_voxel * .5
        new_points = (point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * half_voxel).reshape(-1, 3)
        
        old_coords = (point_xyz / half_voxel).floor_().long()
        new_coords = (old_coords.unsqueeze(1) + offset.unsqueeze(0)).reshape(-1, 3)
        new_keys0 = (new_coords.unsqueeze(1) + offset.unsqueeze(0)).reshape(-1, 3)

        # get unique keys and inverse indices (for original key0, where it maps to in keys)
        new_keys, new_feats = torch.unique(new_keys0, dim=0, sorted=True, return_inverse=True)
        new_keys_idx = new_feats.new_zeros(new_keys.size(0)).scatter_(
            0, new_feats, torch.arange(new_keys0.size(0), device=new_feats.device) // 64)
         
        # recompute key vectors using trilinear interpolation 
        new_feats = new_feats.reshape(-1, 8)
        p = (new_keys - old_coords[new_keys_idx]).type_as(point_xyz).unsqueeze(1) * .25 + 0.5 # (1/4 voxel size)
        q = (self.offset.type_as(p) * .5 + .5).unsqueeze(0)   # (1/2 voxel size)
        new_values = trilinear_interp(p, q, point_feats[new_keys_idx])
        
        # assign to the parameters    
        if update_buffer:
            new_num_keys = new_values.size(0)
            new_point_length = new_points.size(0)

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

        return new_points, new_feats, new_values

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