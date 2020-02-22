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
from torchvision.utils import save_image
from fairdr.data import trajectory, geometry, data_utils

class NeuralRenderer(object):
    
    def __init__(self, resolution=512, frames=300, speed=5, path_gen=None):
        self.frames = frames
        self.speed = speed
        self.path_gen = path_gen
        self.resolution = resolution
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
        
        RT = data_utils.load_matrix("/private/home/jgu/work/tsrn-master/test_trajs/maria/{}.txt".format(t))
        inv_RT = torch.from_numpy(RT).to(intrinsics.device, intrinsics.dtype).inverse()
        
        _, _, cx, cy = geometry.parse_intrinsics(intrinsics)
        cx, cy = int(cx), int(cy)
        v, u = torch.meshgrid([torch.arange(2 * cy), torch.arange(2 * cx)])
        uv = torch.stack([u, v], 0).type_as(intrinsics)
        uv = uv[:, ::2*cy//self.resolution, ::2*cx//self.resolution]
        uv = uv.reshape(2, -1)
        # from fairseq.pdb import set_trace; set_trace()

        ray_start = inv_RT[:3, 3]
        ray_dir = geometry.get_ray_direction(ray_start, uv, intrinsics, inv_RT)
        return ray_start[None, :], ray_dir.transpose(0, 1)

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]
        rgb_path = "/private/home/jgu/data/test_images"
        for shape in range(sample['shape'].size(0)):
            for step in range(self.frames):
                print(step)
                ray_start, ray_dir = self.generate_rays(step, sample['intrinsics'][shape])
                _sample = {
                    'ray_start': ray_start[None, None, :, :],
                    'ray_dir': ray_dir[None, None, :, :],
                    'shape': sample['shape'][0:1], 
                    'view': torch.ones_like(sample['shape'][None, 0:1]) * step
                }
                images = model.visualize(_sample, i=-1)
                
                for key in images:
                    if 'rgb' in key:
                        
                        rgb_name = "{}_{:04d}".format(shape, step)
                        save_image(images[key].permute(2, 0, 1), "{}/rgb/{}.png".format(rgb_path, rgb_name), format=None)
            
            # save as gif
            os.system("ffmpeg -framerate 24 -i {}/rgb/{}_%04d.png -y {}/rgb_512.gif".format(rgb_path, shape, rgb_path))
