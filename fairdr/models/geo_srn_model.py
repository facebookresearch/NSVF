# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairdr.modules.pointnet2.pointnet2_utils import ball_ray_intersect
from fairdr.data.geometry import ray
from fairdr.models.point_srn_model import (
    PointSRNModel, PointSRNField, transformer_base_architecture
)

@register_model('geo_srn')
class GEOSRNModel(PointSRNModel):

    @classmethod
    def build_field(cls, args):
        return GEOSRNField(args)

    def forward(self, ray_start, ray_dir, points, raymarching_steps=None, **kwargs):
        # get geometry features
        feats, xyz = self.field.get_backbone_features(points)
        from fairseq import pdb; pdb.set_trace()
        # corse ray-intersection
        idx, min_depth, max_depth = ball_ray_intersect(
            ray_start, ray_dir, points, 0.1, 10)
        from fairseq import pdb; pdb.set_trace()


class GEOSRNField(PointSRNField):

    def __init__(self, args):
        super().__init__(args)


@register_model_architecture("geo_srn", "geo_srn")
def geo_base_architecture(args):
    transformer_base_architecture(args)
