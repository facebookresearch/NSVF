# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import os
import torch
import numpy as np
import logging
from torchvision.utils import save_image
from fairdr.data import trajectory, geometry, data_utils


logger = logging.getLogger(__name__)


class NeuralRenderer(object):
    
    def __init__(self, 
        resolution=512, 
        frames=501, 
        speed=5, 
        path_gen=None, 
        beam=10):
        self.frames = frames
        self.speed = speed
        self.path_gen = path_gen
        self.resolution = resolution
        self.beam = beam
        if self.path_gen is None:
            self.path_gen = trajectory.circle()

    def generate_rays(self, t, intrinsics):
        # cam_pos = torch.tensor(self.path_gen(t * self.speed / 180 * np.pi), 
        #                        device=intrinsics.device, dtype=intrinsics.dtype)
        # cam_rot = geometry.look_at_rotation(cam_pos, inverse=True, cv=True)
        
        # inv_RT = cam_pos.new_zeros(4, 4)
        # inv_RT[:3, :3] = cam_rot
        # inv_RT[:3, 3] = cam_pos
        # inv_RT[3, 3] = 1

        # generate ray from a fixed trajectory, for now.        
        #RT = data_utils.load_matrix("/private/home/jgu/work/tsrn-master/test_trajs/maria/{}.txt".format(t))
        RT = data_utils.load_matrix( "/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug/extrinsic/model_{0:03d}.txt".format(t))
        RT = np.concatenate([RT, np.zeros((1, 4))], 0)
        RT[3, 3] = 1
        inv_RT = torch.from_numpy(RT).to(intrinsics.device, intrinsics.dtype).inverse()

        _, _, cx, cy = geometry.parse_intrinsics(intrinsics)
        cx, cy = int(cx), int(cy)
        v, u = torch.meshgrid([torch.arange(2 * cy), torch.arange(2 * cx)])
        uv = torch.stack([u, v], 0).type_as(intrinsics)
        uv = uv[:, ::2*cy//self.resolution, ::2*cx//self.resolution]
        uv = uv.reshape(2, -1)
    
        ray_start = inv_RT[:3, 3]
        ray_dir = geometry.get_ray_direction(ray_start, uv, intrinsics, inv_RT)
        return ray_start[None, :], ray_dir.transpose(0, 1)

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]
        rgb_path = "/private/home/jgu/data/test_images"
        for shape in range(sample['shape'].size(0)):
            for step in range(0, self.frames, self.beam):
                logger.info("rendering frames: {}".format(step))
                ray_start, ray_dir = zip(*[
                    self.generate_rays(k, sample['intrinsics'][shape])
                    for k in range(step, min(self.frames, step + self.beam))
                ])

                _sample = {
                    'ray_start': torch.stack(ray_start, 0).unsqueeze(0),
                    'ray_dir': torch.stack(ray_dir, 0).unsqueeze(0),
                    'shape': sample['shape'],
                    'view': torch.arange(
                        step, min(step + self.beam, self.frames), 
                        device=sample['shape'].device).unsqueeze(0)
                }
                
                _ = model(**_sample)

                for k in range(step, min(self.frames, step + self.beam)):
                    images = model.visualize(_sample, None, shape, k-step, 
                                        target_map=False, depth_map=False)
                    rgb_name = "{}y_{:04d}".format(shape, k)
                    for key in images:
                        if 'rgb' in key:
                            save_image(images[key].permute(2, 0, 1), "{}/rgbz/{}.png".format(rgb_path, rgb_name), format=None)
                
            # save as gif
            os.system("ffmpeg -framerate 60 -i {}/rgbz/{}y_%04d.png -y {}/rgb_slurm3.gif".format(rgb_path, shape, rgb_path))
