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

from fairnr.modules.encoder import MultiSparseVoxelEncoder
from fairnr.modules.field import RaidanceField
from fairnr.modules.renderer import VolumeRenderer
from fairnr.modules.reader import Reader
from fairnr.models.nsvf import NSVFModel, base_architecture


@register_model('multi_nsvf')
class MultiNSVFModel(NSVFModel):

    @classmethod
    def build_encoder(cls, args):
        return MultiSparseVoxelEncoder(args)

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


@register_model_architecture('multi_nsvf', "multi_nsvf_base")
def multi_base_architecture(args):
    base_architecture(args)