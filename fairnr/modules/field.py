# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairnr.modules.implicit import (
    ImplicitField, SignedDistanceField,
    TextureField, DiffusionSpecularField
)
from fairnr.modules.linear import NeRFPosEmbLinear


class Field(nn.Module):
    """
    Abstract class for implicit functions
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError


class BackgroundField(Field):
    """
    Background (we assume a uniform color)
    """
    def __init__(self, args):
        super().__init__(args)

        # TODO: we assume a constant background error everywhere
        bg_color = getattr(args, "transparent_background", "1.0,1.0,1.0")
        bg_color = [float(b) for b in bg_color.split(',')] if isinstance(bg_color, str) else [bg_color]
        if getattr(args, "min_color", -1) == -1:
            bg_color = [b * 2 - 1 for b in bg_color]
        if len(bg_color) == 1:
            bg_color = bg_color + bg_color + bg_color
        self.bg_color = nn.Parameter(
            torch.tensor(bg_color), 
            requires_grad=(
                not getattr(args, "background_stop_gradient", False)
            ))
        self.depth = args.background_depth

    @staticmethod
    def add_args(parser):
        # background color
        parser.add_argument('--background-depth', type=float,
                            help='the depth of background. used for depth visualization')
        parser.add_argument('--background-stop-gradient', action='store_true',
                            help='do not optimize the background color')

    def forward(self, **kwargs):
        return self.bg_color * 0 + 1.0  # force to white


class RaidanceField(Field):
    
    def __init__(self, args, input_dim=256):
        super().__init__(args)

        # additional arguments
        self.chunk_size = getattr(args, "chunk_size", 256) * 256
        self.deterministic_step = getattr(args, "deterministic_step", False)       
        self.add_pos_embed = getattr(args, "add_pos_embed", 0)
        self.disable_raydir = getattr(args, "disable_raydir", False)
        self.raydir_embed_dim = args.raydir_embed_dim if (not self.disable_raydir) else 0

        # background field
        self.bg_color = BackgroundField(args)       
            
        # build networks
        self.feature_field = ImplicitField(
                args, 
                input_dim, 
                args.output_embed_dim, 
                args.feature_embed_dim, 
                args.feature_layers,
                pos_proj=self.add_pos_embed)

        # density predictor
        self.predictor = SignedDistanceField(
                args,
                args.output_embed_dim,
                args.density_embed_dim, 
                recurrent=False)
        
        # texture predictor
        self.renderer = TextureField(
                args,
                args.output_embed_dim + self.raydir_embed_dim,
                args.texture_embed_dim,
                args.texture_layers) \
            if not getattr(args, "saperate_specular", False) \
            else DiffusionSpecularField(
                args,
                args.output_embed_dim,
                args.texture_embed_dim,
                self.raydir_embed_dim,
                args.texture_layers,
                getattr(args, "specular_dropout", 0)
            )

        if not self.disable_raydir:            
            self.raydir_proj = NeRFPosEmbLinear(3, self.raydir_embed_dim, angular=True)

    @staticmethod
    def add_args(parser):
        parser.add_argument('--feature-embed-dim', type=int, metavar='N',
                            help='field hidden dimension for FFN')
        parser.add_argument('--density-embed-dim', type=int, metavar='N', 
                            help='hidden dimension of density prediction'),
        parser.add_argument('--texture-embed-dim', type=int, metavar='N',
                            help='hidden dimension of texture prediction')
        parser.add_argument('--input-embed-dim', type=int, metavar='N',
                            help='number of features for query (in default 3, xyz)')
        parser.add_argument('--output-embed-dim', type=int, metavar='N',
                            help='number of features the field returns')
        parser.add_argument('--raydir-embed-dim', type=int, metavar='N',
                            help='the number of dimension to encode the ray directions')
        parser.add_argument('--disable-raydir', action='store_true', 
                            help='if set, not use view direction as additional inputs')

        parser.add_argument('--feature-layers', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--texture-layers', type=int, metavar='N',
                            help='number of FC layers used to predict colors')        

        # specific parameters
        parser.add_argument('--add-pos-embed', type=int, metavar='N',
                            help='using periodic activation augmentation')
        parser.add_argument('--saperate-specular', action='store_true',
                            help='if set, use a different network to predict specular (must provide raydir)')
        parser.add_argument('--specular-dropout', type=float, metavar='D',
                            help='if large than 0, randomly drop specular during training')

        # backgound parameters
        BackgroundField.add_args(parser)

    def forward(self, inputs, dir=None, features=None, outputs=['sigma', 'texture']):
        _data = {}

        if features is None:
            features = self.feature_field(inputs)
            _data['features'] = features

        if 'sigma' in outputs:
            sigma = self.predictor(features)[0]
            _data['sigma'] = sigma

        if 'texture' in outputs:
            if (dir is not None) and (not self.disable_raydir):
                features = torch.cat([features, self.raydir_proj(dir)], -1)
            texture = self.renderer(features)
            _data['texture'] = texture
        
        return tuple([_data[key] for key in outputs])


