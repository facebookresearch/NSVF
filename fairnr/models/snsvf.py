# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# An alternative version of Neural Sparse Voxel Fields
# We use volume rendering to find the intersection, and do surface rendering for color


from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairnr.models.nsvf import NSVFModel, base_architecture
from fairnr.modules.renderer import DeltaVolumeRenderer


@register_model('snsvf')
class SurfaceNSVFModel(NSVFModel):

    @staticmethod
    def add_args(parser):
        DeltaVolumeRenderer.add_args(parser)
        NSVFModel.add_args(parser)
        
    @classmethod
    def build_raymarcher(cls, args):
        return DeltaVolumeRenderer(args)


@register_model_architecture("snsvf", "snsvf_base")
def snsvf_architecture(args):
    # raymarcher
    args.interp_color = getattr(args, "interp_color", False)
    args.exp_depth = getattr(args, "exp_depth", False)
    base_architecture(args)