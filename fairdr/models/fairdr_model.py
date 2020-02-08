# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various fairdr models.

The basic principle of differentiable rendering is two components:
    -- an encoder or so-called geometric encoder (GE)
    -- an decoder or so-called differentiable ray-marcher (RM)
So it can be composed as a GERM model
"""

import logging
import torch.nn as nn

from fairseq.models import BaseFairseqModel


logger = logging.getLogger(__name__)


class BaseFairDRModel(BaseFairseqModel):
    """Base class"""

    def __init__(self, args, encoder, decoder):
        super().__init__()

        self.args = args
        self.encoder = encoder
        self.decoder = decoder
    
        assert isinstance(self.encoder, FairDREncoder)
        assert isinstance(self.decoder, FairDRDecoder)

    def forward(self, **kwargs):
        # ray-interection
        xyz = self.decoder(**kwargs)
        
        # copy color
        return self.encoder(xyz)


class FairDREncoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, xyz, **kwargs):
        """
        Args:
            xyz: query point in the implicit space
        """
        raise NotImplementedError


class FairDRDecoder(nn.Module):

    def __init__(self, args):
        super().__init__()

        self.args = args

    def forward(self, uv, ray, encoder, **kwargs):
        """
        Args:
            uv: coordinates in the image space
            ray: the directional vector for each uv points
        """
        raise NotImplementedError