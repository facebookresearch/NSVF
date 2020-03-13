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

        if self.path_gen is None:
            self.path_gen = trajectory.circle()
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        if self.output_type is None:
            self.output_type = ["rgb"]

    def generate_rays(self, t, intrinsics):
        cam_pos = torch.tensor(self.path_gen(t * self.speed / 180 * np.pi), 
                    device=intrinsics.device, dtype=intrinsics.dtype)
        cam_rot = geometry.look_at_rotation(cam_pos, inverse=True, cv=True)
        
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
        from fairseq import pdb; pdb.set_trace()
        rgb_path = tempfile.mkdtemp()
        for shape in range(sample['shape'].size(0)):
            for step in range(0, self.frames, self.beam):
                logger.info("rendering frames: {}".format(step))
                ray_start, ray_dir, inv_RT = zip(*[
                    self.generate_rays(k, sample['intrinsics'][shape])
                    for k in range(step, min(self.frames, step + self.beam))
                ])

                _sample = {
                    'ray_start': torch.stack(ray_start, 0).unsqueeze(0),
                    'ray_dir': torch.stack(ray_dir, 0).unsqueeze(0),
                    'extrinsics': torch.stack(inv_RT, 0).unsqueeze(0),
                    'shape': sample['shape'],
                    'view': torch.arange(
                        step, min(step + self.beam, self.frames), 
                        device=sample['shape'].device).unsqueeze(0),
                    'voxels': sample.get('voxels', None).clone(),
                    'points': sample.get('points', None).clone(),
                    'raymarching_steps': self.raymarching_steps
                }
                _ = model(**_sample)

                for k in range(step, min(self.frames, step + self.beam)):
                    images = model.visualize(
                                _sample, None, shape, k-step, 
                                target_map=False, 
                                depth_map=('depth' in self.output_type),
                                normal_map=('normal' in self.output_type),
                                hit_map=True)
                    rgb_name = "{}_{:04d}".format(shape, k)
                    
                    for key in images:
                        type = key.split('/')[0]
                        if type in self.output_type:
                            image = images[key].permute(2, 0, 1) \
                                if images[key].dim() == 3 else torch.stack(3*[images[key]], 0)
                            
                            save_image(image, "{}/{}_{}.png".format(rgb_path, type, rgb_name), format=None)
            
            # -- output as gif
            timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
            if self.raymarching_steps is not None:
                timestamp = '_ray{}_'.format(self.raymarching_steps) + timestamp
                
            for type in self.output_type:
                os.system("ffmpeg -framerate 60 -i {}/{}_{}_%04d.png -y {}/{}_{}.gif".format(
                    rgb_path, type, shape, self.output_dir, type, timestamp))
