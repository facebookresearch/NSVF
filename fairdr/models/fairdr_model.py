# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various  models.

The basic principle of differentiable rendering is two components:
    -- an field or so-called geometric field (GE)
    -- an raymarcher or so-called differentiable ray-marcher (RM)
So it can be composed as a GERM model
"""

import logging
import torch.nn as nn

from fairseq.models import BaseFairseqModel


logger = logging.getLogger(__name__)


class BaseModel(BaseFairseqModel):
    """Base class"""

    def __init__(self, args, field, raymarcher):
        super().__init__()
        self.args = args
        self.field = field
        self.raymarcher = raymarcher
        self.cache = None
    
        assert isinstance(self.field, Field)
        assert isinstance(self.raymarcher, Raymarcher)

    def forward(self, **kwargs):
        raise NotImplementedError

    def visualize(self, **kwargs):
        return NotImplementedError

    def pruning(self, **kwargs):
        # do nothing
        pass

class Field(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, xyz, **kwargs):
        raise NotImplementedError


class Raymarcher(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, uv, ray, **kwargs):
        raise NotImplementedError