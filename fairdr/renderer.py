# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import torch

import fairdr.data.trajectory as trajectory

class NeuralRenderer(object):
    
    def __init__(self, frames=300, speed=5, path_gen=None):
        self.frames = frames
        self.speed = speed
        self.path_gen = path_gen
        if self.path_gen is None:
            self.path_gen = trajectory.circle()

    def generate_rays(self, t, intrinsics):
        ray_start = self.path_gen(t)


    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]


        from fairseq.pdb import set_trace; set_trace()