# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairdr.data.geometry import ray
from fairdr.modules.linear import Linear, PosEmbLinear, NeRFPosEmbLinear
from fairdr.modules.implicit import ImplicitField, SignedDistanceField, TextureField, DiffusionSpecularField
from fairdr.models.srn_model import SRNModel, SRNField, base_architecture
from fairdr.modules.backbone import BACKBONE_REGISTRY
from fairdr.modules.pointnet2.pointnet2_utils import ball_nearest

@register_model('point_srn')
class PointSRNModel(SRNModel):

    @classmethod
    def build_field(cls, args):
        return PointSRNField(args)

    @staticmethod
    def add_args(parser):
        SRNModel.add_args(parser)
        
        for backbone in BACKBONE_REGISTRY:
            BACKBONE_REGISTRY[backbone].add_args(parser)

        parser.add_argument("--ball-radius", type=float, metavar='D', 
                            help="maximum radius of ball query for ray-marching")
        parser.add_argument("--backbone", choices=['pointnet2', 'sparseconv', 'transformer'], type=str,
                            help="backbone network, encoding features for the input points")
        parser.add_argument("--relative-position", action='store_true')
        parser.add_argument("--pos-embed", action='store_true', 
                            help='use positional embedding instead of linear projection')
        parser.add_argument("--nerf-pos", action='store_true', 
                            help='instead of standard transformer-based position. use NERF based.')
        parser.add_argument('--use-raydir', action='store_true', 
                            help='if set, use view direction as additional inputs')
        parser.add_argument('--raydir-features', type=int, metavar='N',
                            help='the number of dimension to encode the ray directions')
        parser.add_argument('--saperate-specular', action='store_true',
                            help='if set, use a different network to predict specular (must provide raydir)')
        parser.add_argument('--specular-dropout', type=float, metavar='D',
                            help='if large than 0, randomly drop specular during training')
        parser.add_argument('--raypos-features', type=int, metavar='N', 
                            help='additional to backbone, additional feature dimensions')

    def forward(self, ray_start, ray_dir, points, raymarching_steps=None, **kwargs):
        # get geometry features
        feats, xyz = self.field.get_backbone_features(points)

        # ray intersection
        depths, _ = self.raymarcher(
            self.field.get_sdf, 
            ray_start, ray_dir,
            state=(feats, xyz, None),
            steps=self.args.raymarching_steps 
                if raymarching_steps is None else raymarching_steps)
        points = ray(ray_start, ray_dir, depths.unsqueeze(-1))

        # rendering
        predicts, query_inds, _ = self.field(points, feats, xyz)

        # model's output
        results = {
            'predicts': predicts,
            'depths': depths,
            'grad_penalty': 0,
            'hits': query_inds
        }

        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results


class PointSRNField(SRNField):

    def __init__(self, args):
        SRNField.__init__(self, args)
        try:
            self.backbone = BACKBONE_REGISTRY[args.backbone](args)
        except Exception:
            raise NotImplementedError("Backbone is not implemented!!!")

        self.relative_position = getattr(args, "relative_position", False)
        self.ball_radius = args.ball_radius
        self.use_raydir = getattr(args, "use_raydir", False)
        self.raydir_features = 0
        self.raypos_features = self.backbone.feature_dim \
            if getattr(args, "raypos_features", None) is None \
            else getattr(args, "raypos_features")

        if self.raypos_features > 0:
            if not getattr(args, "pos_embed", False):
                self.point_proj = Linear(args.input_features, self.raypos_features)
            elif not getattr(args, "nerf_pos", False):
                self.point_proj = PosEmbLinear(args.input_features, self.raypos_features)
            else:
                self.point_proj = NeRFPosEmbLinear(args.input_features, self.raypos_features, angular=False)

        if self.use_raydir:
            self.raydir_features = getattr(args, "raydir_features", 144)
            
            if not getattr(args, "pos_embed", False):
                self.raydir_proj = Linear(3, self.raydir_features)
            elif not getattr(args, "nerf_pos", False):
                self.raydir_proj = PosEmbLinear(3, self.raydir_features)
            else:
                self.raydir_proj = NeRFPosEmbLinear(3, self.raydir_features, angular=True)

        self.feature_field = ImplicitField(
            args, 
            self.backbone.feature_dim + self.raypos_features, 
            args.output_features, 
            args.hidden_features, 
            args.num_layer_features - 1)
        self.signed_distance_field = SignedDistanceField(
            args,
            args.output_features,
            args.hidden_sdf,
            args.lstm_sdf)

        if not getattr(args, "saperate_specular", False):
            self.renderer = TextureField(
                args,
                args.output_features + self.raydir_features,
                args.hidden_textures,
                args.num_layer_textures)
        else:
            assert self.use_raydir, "only support view dependent"
            self.renderer = DiffusionSpecularField(
                args,
                args.output_features,
                args.hidden_textures,
                self.raydir_features,
                args.num_layer_textures,
                getattr(args, "specular_dropout", 0)
            )


    def get_backbone_features(self, points):
        return self.backbone(points, add_dummy=True)
        
    def get_feature(self, xyz, point_feats, point_xyz):
        S, V, P, _ = xyz.size()
        # from fairseq import pdb; pdb.set_trace()
        # xyz[:, :, :, 0] += 0.5
        query_inds, query_dis = ball_nearest(self.ball_radius, point_xyz, xyz.view(S, V * P, 3))
        query_inds = query_inds.long()        
        
        if not self.relative_position:
            # if no relative-position, the first feature is a dummy vector
            query_feats = point_feats.gather(
                1, query_inds.unsqueeze(-1).expand(S, V * P, self.backbone.feature_dim) + 1)
        else:
            query_feats = point_feats.gather(
                1, query_inds.unsqueeze(-1).expand(S, V * P, self.backbone.feature_dim))
            query_xyz = point_xyz.gather(
                1, query_inds.unsqueeze(-1).expand(S, V * P, 3)).view(S, V, P, -1)
            xyz = xyz - query_xyz

        input_feats = torch.cat([query_feats.view(S, V, P, -1), self.point_proj(xyz)], -1)
        return self.feature_field(input_feats), query_inds.view(S, V, P), query_dis.view(S, V, P)

    def get_sdf(self, xyz, state=None):
        point_feats, point_xyz, hidden_state = state
        output_feature = self.get_feature(xyz, point_feats, point_xyz)[0]
        depth, hidden_state = self.signed_distance_field(output_feature, hidden_state)
        return depth, (point_feats, point_xyz, hidden_state)

    def get_texture(self, xyz, point_feats, point_xyz, dir=None):
        features, inds, dis = self.get_feature(xyz, point_feats, point_xyz)
        if dir is not None and self.use_raydir:
            features = torch.cat([features, self.raydir_proj(dir)], -1)
        return self.renderer(features), inds, dis

    def forward(self, xyz, point_feats, point_xyz, dir=None):
        return self.get_texture(xyz, point_feats, point_xyz, dir)


def base_point_architecture(args):
    args.pos_embed = getattr(args, "pos_embed", False)
    args.nerf_pos = getattr(args, "nerf_pos", False)
    args.use_raydir = getattr(args, "use_raydir", False)
    args.raydir_features = getattr(args, "raydir_features", 144)
    args.raypos_features = getattr(args, "raypos_features", None)
    args.saperate_specular = getattr(args, "saperate_specular", False)
    args.specular_dropout = getattr(args, "specular_dropout", 0.0)
    args.relative_position = getattr(args, "relative_position", False)
    args.ball_radius = getattr(args, "ball_radius", 0.25)
    base_architecture(args)


@register_model_architecture("point_srn", "embedding_srn")
def embedding_base_architecture(args):
    args.backbone = "embedding"
    args.quantized_voxel_path = getattr(args, "quantized_voxel_path", None)
    args.quantized_embed_dim = getattr(args, "quantized_embed_dim", 256)
    args.quantized_subsampling_points = getattr(args, "quantized_subsampling_points", 256)
    args.quantized_input_shuffle = getattr(args, "quantized_input_shuffle", True)
    args.quantized_voxel_vertex = getattr(args, "quantized_voxel_vertex", False)
    base_point_architecture(args)


@register_model_architecture("point_srn", "dynamic_srn")
def dynamic_embed_base_architecture(args):
    args.backbone = "dynamic_embedding"
    args.quantized_voxel_path = getattr(args, "quantized_voxel_path", None)
    args.quantized_embed_dim = getattr(args, "quantized_embed_dim", 256)
    base_point_architecture(args)


@register_model_architecture("point_srn", "pointnet2_srn")
def pointnet_base_architecture(args):
    args.backbone = "pointnet2"
    args.pointnet2_input_shuffle = getattr(args, "pointnet2_input_shuffle", False)
    args.pointnet2_input_feature_dim = getattr(args, "pointnet2_input_feature_dim", 0)
    args.pointnet2_min_radius = getattr(args, "pointnet2_min_radius", 0.1)
    args.pointnet2_upsample512 = getattr(args, "pointnet2_upsample512", False)
    base_point_architecture(args)


@register_model_architecture("point_srn", "transformer_srn")
def transformer_base_architecture(args):
    args.backbone = "transformer"
    args.subsampling_points = getattr(args, "subsampling_points", 256)
    args.transformer_input_shuffle = getattr(args, "transformer_input_shuffle", False)
    args.furthest_sampling = getattr(args, "furthest_sampling", True)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 3)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.0)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.transformer_pos_embed = getattr(args, "transformer_pos_embed", False)
    base_point_architecture(args)