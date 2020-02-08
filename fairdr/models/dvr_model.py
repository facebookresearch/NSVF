# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a re-implementation of 
"Differentiable Volumetric Rendering: 
Learning Implicit 3D Representations without 3D Supervision"
https://arxiv.org/pdf/1912.07372.pdf
"""

import torch.nn as nn

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairdr.models.fairdr_model import BaseFairDRModel, FairDREncoder, FairDRDecoder
from fairdr.modules.implicit import ImplicitField
from fairdr.modules.raymarcher import UniformSearchRayMarcher

@register_model('diffentiable_volumetric_rendering')
class DVRModel(BaseFairDRModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        base_architecture(args)

        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args)
        return cls(args, encoder, decoder)
    
    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--ffn-embed-dim', type=int, metavar='N',
                            help='encoder input dimension for FFN')
        parser.add_argument('--ffn-hidden-dim', type=int, metavar='N',
                            help='encoder hidden dimension for FFN')
        parser.add_argument('--input-features', type=int, metavar='N',
                            help='number of features for query')
        parser.add_argument('--output-features', type=int, metavar='N',
                            help='number of features the field returns')
        parser.add_argument('--ffn-num-layers', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--use-residual', action='store_true')


    @classmethod
    def build_encoder(cls, args):
        return DVREncoder(args)

    @classmethod
    def build_decoder(cls, args):
        return DVRDecoder(args)

    def forward(self, ray_start, ray_dir, rgb, alpha, **kwargs):
        from fairseq.pdb import set_trace; set_trace()

        pass


class DVREncoder(FairDREncoder):

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.field = ImplicitField(args)  

    def occupancy(self, xyz):
        return self.field(xyz)[1][:, :, :, -1]

    def forward(self, xyz):
        """
        xyz: shape x view x pixel x world_coords
        """
        return self.field(xyz)[1][:, :, :, :3]  # rgb


class DVRDecoder(FairDRDecoder):

    def __init__(self, args):
        super().__init__(args)

        self.args = args
        self.raymarcher = UniformSearchRayMarcher(args) 




@register_model_architecture("diffentiable_volumetric_rendering", "dvr_base")
def base_architecture(args):
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim",    256)
    args.ffn_hidden_dim = getattr(args, "ffn_hidden_dim",  256)
    args.ffn_num_layers = getattr(args, "ffn_num_layers",  3)
    args.input_features = getattr(args, "input_features",  3)   # xyz
    args.output_features =getattr(args, "output_features", 4)  # texture (3) + Occupancy(1)
    args.use_residual = getattr(args, "use_residual", True)