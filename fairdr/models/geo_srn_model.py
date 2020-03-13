# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairdr.data.geometry import ray
from fairdr.models.srn_model import SRNModel, SRNField, base_architecture

@register_model('geo_srn')
class GEOSRNModel(SRNModel):

    @classmethod
    def build_field(cls, args):
        return GEOSRNField(args)

    def forward(self, ray_start, ray_dir, raymarching_steps=None, **kwargs):
        from fairseq import pdb; pdb.set_trace()


class GEOSRNField(SRNField):

    def __init__(self, args):
        super().__init__(args)


@register_model_architecture("geo_srn", "geo_srn")
def geo_base_architecture(args):
    base_architecture(args)
