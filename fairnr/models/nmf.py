# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
logger = logging.getLogger(__name__)

import torch
from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairnr.models.nsvf import NSVFModel


@register_model('nmf')
class NMFModel(NSVFModel):
    """
    Experimental code: Neural Mesh Field
    """
    ENCODER = 'triangle_mesh_encoder'

    @torch.no_grad()
    def prune_voxels(self, *args, **kwargs):
        pass
    
    @torch.no_grad()
    def split_voxels(self):
        pass
        # logger.info("half the global cage size {:.4f} -> {:.4f}".format(
        #     self.encoder.cage_size.item(), self.encoder.cage_size.item() * .5))
        # self.encoder.cage_size *= .5


@register_model_architecture("nmf", "nmf_base")
def base_architecture(args):
    # parameter needs to be changed
    args.max_hits = getattr(args, "max_hits", 60)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
    
    # encoder default parameter
    args.voxel_embed_dim = getattr(args, "voxel_embed_dim", 0)
    args.voxel_path = getattr(args, "voxel_path", None)

    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "pos:10")
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, pos:10, ray:4")
    args.feature_embed_dim = getattr(args, "feature_embed_dim", 256)
    args.density_embed_dim = getattr(args, "density_embed_dim", 128)
    args.texture_embed_dim = getattr(args, "texture_embed_dim", 256)

    args.feature_layers = getattr(args, "feature_layers", 1)
    args.texture_layers = getattr(args, "texture_layers", 3)
    
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.background_depth = getattr(args, "background_depth", 5.0)
    
    # raymarcher
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)
    args.raymarching_tolerance = getattr(args, "raymarching_tolerance", 0)

    # reader
    args.pixel_per_view = getattr(args, "pixel_per_view", 2048)
    args.sampling_on_mask = getattr(args, "sampling_on_mask", 0.0)
    args.sampling_at_center = getattr(args, "sampling_at_center", 1.0)
    args.sampling_on_bbox = getattr(args, "sampling_on_bbox", False)
    args.sampling_patch_size = getattr(args, "sampling_patch_size", 1)
    args.sampling_skipping_size = getattr(args, "sampling_skipping_size", 1)

    # others
    args.chunk_size = getattr(args, "chunk_size", 64)


@register_model_architecture("nmf", "nmf_nerf")
def nerf_style_architecture(args):
    args.inputs_to_texture = getattr(args, "inputs_to_texture", "feat:0:256, ray:4")
    args.feature_layers = getattr(args, "feature_layers", 6)
    args.texture_layers = getattr(args, "texture_layers", 0)
    args.feature_field_skip_connect = getattr(args, "feature_field_skip_connect", 3)
    args.no_layernorm_mlp = getattr(args, "no_layernorm_mlp", True)
    base_architecture(args)