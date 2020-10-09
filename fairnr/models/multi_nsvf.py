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
from fairnr.models.nsvf import NSVFModel, base_architecture


@register_model('multi_nsvf')
class MultiNSVFModel(NSVFModel):

    ENCODER = 'multi_sparsevoxel_encoder'

    @torch.no_grad()
    def split_voxels(self):
        logger.info("half the global voxel size {:.4f} -> {:.4f}".format(
            self.encoder.all_voxels[0].voxel_size.item(), 
            self.encoder.all_voxels[0].voxel_size.item() * .5))
        self.encoder.splitting()
        for id in range(len(self.encoder.all_voxels)):
            self.encoder.all_voxels[id].voxel_size *= .5
            self.encoder.all_voxels[id].max_hits *= 1.5
        
    @torch.no_grad()
    def reduce_stepsize(self):
        logger.info("reduce the raymarching step size {:.4f} -> {:.4f}".format(
            self.encoder.all_voxels[0].step_size.item(), 
            self.encoder.all_voxels[0].step_size.item() * .5))
        for id in range(len(self.encoder.all_voxels)):
            self.encoder.all_voxels[id].step_size *= .5


@register_model("shared_nsvf")
class SharedNSVFModel(MultiNSVFModel):

    ENCODER = 'shared_sparsevoxel_encoder'
    

@register_model_architecture('multi_nsvf', "multi_nsvf_base")
def multi_base_architecture(args):
    base_architecture(args)


@register_model_architecture('shared_nsvf', 'shared_nsvf')
def shared_base_architecture(args):
    # encoder
    args.context_embed_dim = getattr(args, "context_embed_dim", 96)
    
    # field
    args.inputs_to_density = getattr(args, "inputs_to_density", "emb:6:32, context:0:96")
    args.hypernetwork = getattr(args, "hypernetwork", False)
    base_architecture(args)