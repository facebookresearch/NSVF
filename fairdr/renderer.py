# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import os, tempfile
import time
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
                raymarching_steps=None,
                path_gen=None, 
                beam=10,
                at=(0,0,0),
                up=(0,1,0),
                output_dir=None,
                output_type=None):

        self.frames = frames
        self.speed = speed
        self.raymarching_steps = raymarching_steps
        self.path_gen = path_gen
        self.resolution = resolution
        self.beam = beam
        self.output_dir = output_dir
        self.output_type = output_type
        self.at = at
        self.up = up
    
        if self.path_gen is None:
            self.path_gen = trajectory.circle()
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.output_type is None:
            self.output_type = ["rgb"]

    def generate_rays(self, t, intrinsics):
        cam_pos = torch.tensor(self.path_gen(t * self.speed / 180 * np.pi), 
                    device=intrinsics.device, dtype=intrinsics.dtype)
        cam_rot = geometry.look_at_rotation(cam_pos, at=self.at, up=self.up, inverse=True, cv=True)
        
        inv_RT = cam_pos.new_zeros(4, 4)
        inv_RT[:3, :3] = cam_rot
        inv_RT[:3, 3] = cam_pos
        inv_RT[3, 3] = 1

        # -- generate ray from a fixed trajectory --      
        # RT = data_utils.load_matrix("/private/home/jgu/work/tsrn-master/test_trajs/maria/{}.txt".format(t))
        # RT = data_utils.load_matrix( "/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug/extrinsic/model_{0:03d}.txt".format(t))
        # RT = np.concatenate([RT, np.zeros((1, 4))], 0)
        # RT[3, 3] = 1
        # inv_RT = torch.from_numpy(RT).to(intrinsics.device, intrinsics.dtype).inverse()

        _, _, cx, cy = geometry.parse_intrinsics(intrinsics)
        cx, cy = int(cx), int(cy)
        v, u = torch.meshgrid([torch.arange(2 * cy), torch.arange(2 * cx)])
        uv = torch.stack([u, v], 0).type_as(intrinsics)
        uv = uv[:, ::2*cy//self.resolution, ::2*cx//self.resolution]
        uv = uv.reshape(2, -1)
    
        ray_start = inv_RT[:3, 3]
        ray_dir = geometry.get_ray_direction(ray_start, uv, intrinsics, inv_RT)
        return ray_start[None, :], ray_dir.transpose(0, 1), inv_RT

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]
        model.eval()
        
        rgb_path = tempfile.mkdtemp()
        image_names = []
        sample, step = sample

        for shape in range(sample['shape'].size(0)):
            logger.info("rendering frames: {}".format(step))
            ray_start, ray_dir, inv_RT = zip(*[
                self.generate_rays(k, sample['intrinsics'][shape])
                for k in range(step, step + self.beam)
            ])
        
            voxels, points = sample.get('voxels', None), sample.get('points', None)
            _sample = {
                'ray_start': torch.stack(ray_start, 0).unsqueeze(0),
                'ray_dir': torch.stack(ray_dir, 0).unsqueeze(0),
                'extrinsics': torch.stack(inv_RT, 0).unsqueeze(0),
                'shape': sample['shape'][shape:shape+1],
                'view': torch.arange(
                    step, min(step + self.beam, self.frames), 
                    device=sample['shape'].device).unsqueeze(0),
                'voxels': voxels[shape:shape+1].clone() if voxels is not None else None,
                'points': points[shape:shape+1].clone() if points is not None else None,
                'raymarching_steps': self.raymarching_steps
            }
            _ = model(**_sample)
            # from fairseq import pdb; pdb.set_trace()
            for k in range(step, step + self.beam):
                images = model.visualize(
                            _sample, None, 0, k-step, 
                            target_map=False, 
                            depth_map=('depth' in self.output_type),
                            normal_map=('normal' in self.output_type),
                            hit_map=True)
                rgb_name = "{:04d}".format(k)
                
                for key in images:
                    type = key.split('/')[0]
                    if type in self.output_type:
                        image = images[key].permute(2, 0, 1) \
                            if images[key].dim() == 3 else torch.stack(3*[images[key]], 0)
                        image_name = "{}/{}_{}.png".format(rgb_path, type, rgb_name)
                        save_image(image, image_name, format=None)
                        image_names.append(image_name)
            step = step + self.beam
        
        return step, image_names

