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
import MinkowskiEngine as ME

from fairdr.modules.pointnet2.pointnet2_modules import (
    PointnetSAModuleVotes, PointnetFPModule
)
from fairdr.data.data_utils import load_matrix, unique_points
from fairdr.data.geometry import trilinear_interp
from fairdr.modules.pointnet2.pointnet2_utils import furthest_point_sample
from fairdr.modules.linear import FCBlock, Linear, Embedding
from fairdr.modules.hyper import HyperFC
from fairdr.modules.implicit import SignedDistanceField as SDF
from fairdr.modules.me import unet
from fairseq import options, utils
from fairseq.modules import (
    SinusoidalPositionalEmbedding,
    LayerNorm, MultiheadAttention
)


BACKBONE_REGISTRY = {}
INF = 1000.0

def register_backnone(name):
    def register_backbone_cls(cls):
        if name in BACKBONE_REGISTRY:
            raise ValueError('Cannot register duplicate backbone module ({})'.format(name))
        BACKBONE_REGISTRY[name] = cls
        return cls
    return register_backbone_cls


def padding_points(xs, pad):
    if len(xs) == 1:
        return xs[0].unsqueeze(0)
    
    maxlen = max([x.size(0) for x in xs])
    xt = xs[0].new_ones(len(xs), maxlen, xs[0].size(1)).fill_(pad)
    for i in range(len(xs)):
        xt[i, :xs[i].size(0)] = xs[i]
    return xt


def pruning_points(feats, points, alpha):
    feats = [feats[i][alpha[i]] for i in range(alpha.size(0))]
    points = [points[i][alpha[i]] for i in range(alpha.size(0))]
    points = padding_points(points, INF)
    feats = padding_points(feats, 0)
    return feats, points


def splitting_points(point_xyz, point_feats, offset, half_voxel):        
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
    q = (offset.type_as(p) * .5 + .5).unsqueeze(0)   # (1/2 voxel size)
    new_values = trilinear_interp(p, q, point_feats[new_keys_idx])

    return new_points, new_feats, new_values


def positional_encoding(out_dim, x):
    half_dim = out_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
    x = x.unsqueeze(-1) @ emb.unsqueeze(0)
    x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
    x = x.view(x.size(0), -1)
    return x


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


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

    def latent_regularization(self):
        return 0

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

        self.voxel_size = args.voxel_size  # voxel size
        self.march_size = args.raymarching_stepsize   # raymarching step size (used for pruning)

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
        self.use_pos_embed = getattr(args, "quantized_pos_embed", False)
        self.use_xyz_embed = getattr(args, "quantized_xyz_embed", False)
        self.use_context_proj = getattr(args, "quantized_context_proj", False)
        self.use_hypernetwork = getattr(args, "use_hypernetwork", False)
        self.post_context = getattr(args, "post_context", False)
        self.normalize_context = getattr(args, "normalize_context", False)

        if self.use_context is not None:
            if self.use_context == 'id':
                assert self.args.total_num_context > 0, "index embeddings for different frames"
                self.context_embed = Embedding(self.args.total_num_context, self.embed_dim, None)

            if self.normalize_context:
                self.context_proj = FCBlock(self.embed_dim, 1, self.embed_dim, self.embed_dim)
            elif self.use_context_proj:
                self.context_proj = Linear(self.embed_dim, self.embed_dim)
            else:
                self.context_proj = None

        if self.use_pos_embed:
            assert self.embed_dim is not None and self.embed_dim % 3 == 0, "size mismatch!"
            # -- positional embeddings --
            self.values = Embedding(self.total_size, self.embed_dim, None)
            self.values.weight.data = positional_encoding(self.embed_dim // 3, init_keys.float())
            self.values.weight.requires_grad = False

            if self.use_context is not None and self.use_hypernetwork:
                raise NotImplementedError
            else:
                self.values_proj = Linear(self.embed_dim, self.embed_dim)
        
        elif self.use_xyz_embed:
            self.values = Embedding(init_keys.shape[0], 3, None)
            self.values.weight.data = (init_keys.float() / abs(init_keys.float()).max())
            self.values.weight.requires_grad = False

            if self.use_context is not None and self.use_hypernetwork:
                self.values_hyper = HyperFC(
                        hyper_in_ch=self.embed_dim,
                        hyper_num_hidden_layers=1,
                        hyper_hidden_ch=self.embed_dim,
                        hidden_ch=self.embed_dim,
                        num_hidden_layers=1,
                        in_ch=3, out_ch=self.embed_dim,
                        outermost_linear=True)
            else:
                self.values_proj = Linear(3, self.embed_dim)
            
        else:
            self.values = Embedding(self.total_size, self.embed_dim, None)
            if self.use_context is not None and self.use_hypernetwork:
                self.values_hyper = HyperFC(
                        hyper_in_ch=self.embed_dim,
                        hyper_num_hidden_layers=1,
                        hyper_hidden_ch=self.embed_dim,
                        hidden_ch=self.embed_dim,
                        num_hidden_layers=1,
                        in_ch=self.embed_dim, 
                        out_ch=self.embed_dim,
                        outermost_linear=True)
            else:
                self.values_proj = None

    @staticmethod
    def add_args(parser):
        parser.add_argument('--quantized-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--quantized-pproj-dim', type=int, metavar='N', help="only useful if online_pruning set")
        parser.add_argument('--total-num-context', type=int, metavar='N', help="if use id as inputs, unique embeddings")
        parser.add_argument('--quantized-pos-embed', action='store_true', help="instead of standard embeddings, use positional embeddings")
        parser.add_argument('--quantized-xyz-embed', action='store_true', help='instead of positional embedding, use actual xyz as inputs')
        parser.add_argument('--quantized-context-proj', action='store_true')
        parser.add_argument('--use-hypernetwork', action='store_true')
        parser.add_argument('--post-context', action='store_true', help='redo contexturalization every time voxels splitted')
        parser.add_argument('--normalize-context', action='store_true', help='normalize the embedding vectors onto the sphere')

    def contexturalization(self, context, values, keys=None):
        if self.use_hypernetwork:
            values_proj, context = self.values_hyper(context.squeeze(1)), None
        else:
            values_proj = self.values_proj
        
        if values_proj is not None:
            values = values_proj(values)

        if context is not None:
            if self.normalize_context:
                context = context * torch.rsqrt(1e-8 + torch.mean(context ** 2, dim=-1, keepdim=True))
            if self.context_proj is not None:
                context = self.context_proj(context)
        
        return values, context

    def _forward(self, id, step=0, pruner=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        keys   = self.keys[: self.num_keys]
        
        # extend size to support multi-objects
        feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
        points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
        values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous()
        codes = self.get_context_features(id)

        # contexturalization (potential)
        voxel_size, march_size = self.voxel_size, self.march_size
        values, context = self.contexturalization(codes, values, keys)

        def combine(values, context):
            if context is not None:
                if not self.post_context:   
                    out_values = values + context
                else:
                    out_values = torch.cat([values, context], 1)
                return out_values
            return values

        for t in range(step + 1):
            feats = feats + values.size(1) * torch.arange(values.size(0), 
                device=feats.device, dtype=feats.dtype)[:, None, None]
            out_values = combine(values, context)

            if not self.online_pruning:
                break
            
            if t < step:
                # pruning
                with torch.no_grad():
                    predicts = pruner(
                        id=id, update=False, 
                        features=(feats, points, out_values),
                        sizes=(voxel_size, march_size)).detach()
                
                feats, points = pruning_points(feats, points, predicts > 0.5)
                corner_features = self.get_features(feats, values.view(-1, values.size(-1)))  # recompute features
                
                # splitting
                voxel_size = voxel_size * .5
                march_size = march_size * .5
                new_points, new_feats, new_values = zip(*[splitting_points(
                    points[i], corner_features[i], self.offset, voxel_size
                ) for i in range(feats.size(0))])
                
                points = padding_points(new_points, INF)
                feats = padding_points(new_feats, 0)
                values = padding_points(new_values, 0.0)

        return feats, points, out_values, codes
              
    def get_features(self, x, values):
        return F.embedding(x, values)

    def get_context_features(self, id):
        context = None
        if self.use_context is not None:
            if self.use_context == 'id':
                context = self.context_embed(id).unsqueeze(1)
            else:
                raise NotImplementedError("only add-on condition for now.")
        return context

    def pruning(self, keep):
        self.keep.masked_scatter_(self.keep.bool(), keep.long())

    def splitting(self, half_voxel, update_buffer=True, features=None):
        if features is not None:
            point_xyz, feats, values = features
            point_feats = F.embedding(feats, values)
        
        else:
            point_xyz, feats = self.points[self.keep.bool()], self.feats[self.keep.bool()]
            point_feats = self.values(feats)

        # split points
        new_points, new_feats, new_values = splitting_points(point_xyz, point_feats, self.offset, half_voxel)
        
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

    @property
    def networks_not_freeze(self):
        if self.use_context is not None and self.use_context == 'id':
            return ["context_embed.weight"]
        else:
            raise NotImplementedError


@register_backnone("transformer")
class TransformerBackbone(DynamicEmbeddingBackbone):
    
    def __init__(self, args):
        super().__init__(args)

        assert args.quantized_embed_dim == args.encoder_embed_dim, \
            "for now, we only support the same sizes"

        # additional transformer parameters
        self.dropout = args.dropout
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        self.num_layers = len(self.layers)
        self.embed_dim = self.args.encoder_embed_dim
        self.over_residual = self.args.over_residual
        self.attention_context = self.args.attention_context
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None
        
        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        else:
            self.layernorm_embedding = None

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
        parser.add_argument('--over-residual', action='store_true')
        parser.add_argument('--attention-context', action='store_true')
        parser.add_argument('--cross-attention-context', action='store_true')

    def contexturalization(self, context, values, keys=None):
        m0 = values.eq(0.0).all(-1)
        x0, c = super().contexturalization(context, values, keys=None)

        if c is None:
            x = x0
            m = m0
        elif not self.attention_context:
            x = x0 + c
            m = m0
        else:
            x = torch.cat([c, x0], 1)
            m = torch.cat([m0.new_zeros(m0.size(0), 1), m0], 1)
        
        x = x.transpose(0, 1)
        x0 = x0.transpose(0, 1)
        for layer in self.layers:
            x = layer(x, m, x0, m0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)
        x = x.transpose(0, 1)  
        
        if self.attention_context:
            x = x[:, 1:]
        
        if self.over_residual:
            return x.contiguous() + values, None
        return x.contiguous(), None

    @property
    def feature_dim(self):
        return self.args.encoder_embed_dim


@register_backnone("minkunet")
class MinkUNetBackbone(DynamicEmbeddingBackbone):

    def __init__(self, args):
        super().__init__(args)
        self.unet = getattr(unet, args.unet_arch)(
           in_channels=self.embed_dim, out_channels=self.embed_dim, D=3)
        self.unet = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.unet)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument("--unet-arch", type=str)

    def contexturalization(self, id, values, keys=None):
        x = super().contexturalization(id, values, keys=None)

        # build me batches
        coords = [
            torch.cat([keys.new_ones(keys.size(0), 1) * b, keys], 1) 
        for b in range(x.size(0))]
        coords = torch.cat(coords, 0).cpu().int()
        feats = values.reshape(-1, values.size(-1))
        input = ME.SparseTensor(feats, coords=coords)
        
        # execuate U-Net
        output = self.unet(input)
        
        return output.F.reshape(x.size(0), x.size(1), -1)


class TransformerEncoderLayer(nn.Module):
    """Modify the original EncoderLayer
    """

    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiheadAttention(
            self.embed_dim, args.encoder_attention_heads,
            dropout=args.attention_dropout
        )
        self.dropout = args.dropout
        self.relu_dropout = args.activation_dropout
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = nn.Linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = nn.Linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for i in range(2)])

        if getattr(args, "cross_attention_context", False):
            self.cross_attn = MultiheadAttention(
                self.embed_dim, args.encoder_attention_heads,
                dropout=args.attention_dropout
            )
        else:
            self.cross_attn = None

    def forward(self, x, encoder_padding_mask, context=None, context_mask=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x = self.maybe_layer_norm(0, x, before=True)
        x, _ = self.self_attn(query=x, key=x, value=x, key_padding_mask=encoder_padding_mask)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.maybe_layer_norm(0, x, after=True)

        if self.cross_attn is not None:
            residual = x
            x = self.maybe_layer_norm(0, x, before=True)
            x, _ = self.cross_attn(query=x, key=context, value=context, key_padding_mask=context_mask)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = residual + x
            x = self.maybe_layer_norm(0, x, after=True)

        residual = x
        x = self.maybe_layer_norm(1, x, before=True)
        x = gelu(self.fc1(x.transpose(0, 1)))
        x = F.dropout(x, p=self.relu_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x.transpose(0, 1)
        x = self.maybe_layer_norm(1, x, after=True)
        return x

    def maybe_layer_norm(self, i, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return self.layer_norms[i](x)
        else:
            return x