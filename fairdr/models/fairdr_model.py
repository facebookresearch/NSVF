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

from fairseq.models import BaseFairseqModel


logger = logging.getLogger(__name__)


class BaseFairDRModel(BaseFairseqModel):
    """Base class"""

    def __init__(self, encoder, raymarcher):
        super().__init__()

        self.encoder = encoder
        self.raymarcher = raymarcher
    
    