# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import os, tempfile, shutil, glob
import time
import torch
import numpy as np
import logging
import imageio

from torchvision.utils import save_image
from fairnr.data import trajectory, geometry, data_utils
from fairseq.meters import StopwatchMeter
from fairnr.data.data_utils import recover_image, get_uv, parse_views
from pathlib import Path

logger = logging.getLogger(__name__)


class NeuralRenderer(object):
    
    def __init__(self, 
                resolution="512x512", 
                frames=501, 
                speed=5,
                raymarching_steps=None,
                path_gen=None, 
                beam=10,
                at=(0,0,0),
                up=(0,1,0),
                output_dir=None,
                output_type=None,
                fps=24,
                test_camera_poses=None,
                test_camera_intrinsics=None,
                test_camera_views=None):

        self.frames = frames
        self.speed = speed
        self.raymarching_steps = raymarching_steps
        self.path_gen = path_gen
        
        if isinstance(resolution, str):
            self.resolution = [int(r) for r in resolution.split('x')]
        else:
            self.resolution = [resolution, resolution]

        self.beam = beam
        self.output_dir = output_dir
        self.output_type = output_type
        self.at = at
        self.up = up
        self.fps = fps

        if self.path_gen is None:
            self.path_gen = trajectory.circle()
        if self.output_type is None:
            self.output_type = ["rgb"]

        if test_camera_intrinsics is not None:
            self.test_int = data_utils.load_intrinsics(test_camera_intrinsics)
        else:
            self.test_int = None

        self.test_frameids = None
        if test_camera_poses is not None:
            if os.path.isdir(test_camera_poses):
                self.test_poses = [
                    np.loadtxt(f)[None, :, :] for f in sorted(glob.glob(test_camera_poses + "/*.txt"))]
                self.test_poses = np.concatenate(self.test_poses, 0)
            else:
                self.test_poses = data_utils.load_matrix(test_camera_poses)
                if self.test_poses.shape[1] == 17:
                    self.test_frameids = self.test_poses[:, -1].astype(np.int32)
                    self.test_poses = self.test_poses[:, :-1]
                self.test_poses = self.test_poses.reshape(-1, 4, 4)

            if test_camera_views is not None:
                render_views = parse_views(test_camera_views)
                self.test_poses = np.stack([self.test_poses[r] for r in render_views])

        else:
            self.test_poses = None

    def generate_rays(self, t, intrinsics, img_size, inv_RT=None, action='none'):
        if inv_RT is None:
            cam_pos = torch.tensor(self.path_gen(t * self.speed / 180 * np.pi), 
                        device=intrinsics.device, dtype=intrinsics.dtype)
            cam_rot = geometry.look_at_rotation(cam_pos, at=self.at, up=self.up, inverse=True, cv=True)
            
            inv_RT = cam_pos.new_zeros(4, 4)
            inv_RT[:3, :3] = cam_rot
            inv_RT[:3, 3] = cam_pos
            inv_RT[3, 3] = 1
        else:
            inv_RT = torch.from_numpy(inv_RT).type_as(intrinsics)
        
        h, w, rh, rw = img_size[0], img_size[1], img_size[2], img_size[3]
        if self.test_int is not None:
            uv = torch.from_numpy(get_uv(h, w, h, w)[0]).type_as(intrinsics)
            intrinsics = self.test_int
        else:
            uv = torch.from_numpy(get_uv(h * rh, w * rw, h, w)[0]).type_as(intrinsics)

        uv = uv.reshape(2, -1)
        return uv, inv_RT

    def parse_sample(self,sample):
        if len(sample) == 1:
            return sample[0], 0, self.frames
        elif len(sample) == 2:
            return sample[0], sample[1], self.frames
        elif len(sample) == 3:
            return sample[0], sample[1], sample[2]
        else:
            raise NotImplementedError

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]
        model.eval()
        
        logger.info("rendering starts. {}".format(model.text))
        output_path = self.output_dir
        image_names = []
        sample, step, frames = self.parse_sample(sample)

        # fix the rendering size
        a = sample['size'][0,0,0] / self.resolution[0]
        b = sample['size'][0,0,1] / self.resolution[1]
        sample['size'][:, :, 0] /= a
        sample['size'][:, :, 1] /= b
        sample['size'][:, :, 2] *= a
        sample['size'][:, :, 3] *= b

        for shape in range(sample['shape'].size(0)):
            max_step = step + frames
            while step < max_step:
                next_step = min(step + self.beam, max_step)
                uv, inv_RT = zip(*[
                    self.generate_rays(
                        k, 
                        sample['intrinsics'][shape], 
                        sample['size'][shape, 0],
                        self.test_poses[k] if self.test_poses is not None else None)
                    for k in range(step, next_step)
                ])
                if self.test_frameids is not None:
                    assert next_step - step == 1
                    ids = torch.tensor(self.test_frameids[step: next_step]).type_as(sample['id'])
                else:
                    ids = sample['id'][shape:shape+1]

                real_images = sample['full_rgb'] if 'full_rgb' in sample else sample['colors']
                real_images = real_images.transpose(2, 3) if real_images.size(-1) != 3 else real_images
                
                _sample = {
                    'id': ids,
                    'colors': torch.cat([real_images[shape:shape+1] for _ in range(step, next_step)], 1),
                    'intrinsics': sample['intrinsics'][shape:shape+1],
                    'extrinsics': torch.stack(inv_RT, 0).unsqueeze(0),
                    'uv': torch.stack(uv, 0).unsqueeze(0),
                    'shape': sample['shape'][shape:shape+1],
                    'view': torch.arange(
                        step, next_step, 
                        device=sample['shape'].device).unsqueeze(0),
                    'size': torch.cat([sample['size'][shape:shape+1] for _ in range(step, next_step)], 1),
                    'step': step
                }
                with data_utils.GPUTimer() as timer:
                    outs = model(**_sample)
                logger.info("rendering frame={}\ttotal time={:.4f}".format(step, timer.sum))

                for k in range(step, next_step):
                    images = model.visualize(_sample, None, 0, k-step)
                    image_name = "{:04d}".format(k)

                    for key in images:
                        name, type = key.split('/')[0].split('_')
                        if type in self.output_type:
                            if name == 'coarse':
                                type = 'coarse-' + type
                            if name == 'target':
                                continue
                            
                            prefix = os.path.join(output_path, type)
                            Path(prefix).mkdir(parents=True, exist_ok=True)
                            if type == 'point':
                                data_utils.save_point_cloud(
                                    os.path.join(prefix, image_name + '.ply'),
                                    images[key][:, :3].cpu().numpy(), 
                                    (images[key][:, 3:] * 255).cpu().int().numpy())
                                # from fairseq import pdb; pdb.set_trace()

                            else:
                                image = images[key].permute(2, 0, 1) \
                                    if images[key].dim() == 3 else torch.stack(3*[images[key]], 0)        
                                save_image(image, os.path.join(prefix, image_name + '.png'), format=None)
                                image_names.append(os.path.join(prefix, image_name + '.png'))
                    
                    # save pose matrix
                    prefix = os.path.join(output_path, 'pose')
                    Path(prefix).mkdir(parents=True, exist_ok=True)
                    pose = self.test_poses[k] if self.test_poses is not None else inv_RT[k-step].cpu().numpy()
                    np.savetxt(os.path.join(prefix, image_name + '.txt'), pose)    

                step = next_step

        logger.info("done")
        return step, image_names

    def save_images(self, output_files, steps=None, combine_output=True):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
        if steps is not None:
            timestamp = "step_{}.".format(steps) + timestamp
        
        if not combine_output:
            for type in self.output_type:
                images = [imageio.imread(file_path) for file_path in output_files if type in file_path] 
                # imageio.mimsave('{}/{}_{}.gif'.format(self.output_dir, type, timestamp), images, fps=self.fps)
                imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, type, timestamp), images, fps=self.fps, quality=8)
        else:
            images = [[imageio.imread(file_path) for file_path in output_files if type == file_path.split('/')[-2]] for type in self.output_type]
            images = [np.concatenate([images[j][i] for j in range(len(images))], 1) for i in range(len(images[0]))]
            imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, 'full', timestamp), images, fps=self.fps, quality=8)
        
        return timestamp

    def merge_videos(self, timestamps):
        logger.info("mergining mp4 files..")
        timestamp = time.strftime('%Y-%m-%d.%H-%M-%S',time.localtime(time.time()))
        writer = imageio.get_writer(
            os.path.join(self.output_dir, 'full_' + timestamp + '.mp4'), fps=self.fps)
        for timestamp in timestamps:
            tempfile = os.path.join(self.output_dir, 'full_' + timestamp + '.mp4')
            reader = imageio.get_reader(tempfile)
            for im in reader:
                writer.append_data(im)
        writer.close()
        for timestamp in timestamps:
            tempfile = os.path.join(self.output_dir, 'full_' + timestamp + '.mp4')
            os.remove(tempfile)