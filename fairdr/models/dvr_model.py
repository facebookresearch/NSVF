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
                            help='encoder embedding dimension for FFN')

    @classmethod
    def build_encoder(cls, args):
        return DVREncoder(args)

    @classmethod
    def build_decoder(cls, args):
        return FairDRDecoder(args)


class DVREncoder(FairDREncoder):

    def __init__(self, args):
        super().__init__(args)

        self.fc = nn.Linear(args.ffn_embed_dim, args.ffn_embed_dim)
        


@register_model_architecture("diffentiable_volumetric_rendering", "dvr_base")
def base_architecture(args):
    args.ffn_embed_dim = getattr(args, "ffn_embed_dim", 256)