# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file is to simulate "generator" in fairseq
"""

import os, tempfile, shutil
import time
import torch
import numpy as np
import logging
import imageio

from torchvision.utils import save_image
from fairdr.data import trajectory, geometry, data_utils
from fairseq.meters import StopwatchMeter
from fairdr.data.data_utils import recover_image, get_uv

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
                output_type=None,
                fps=24,
                test_camera_poses=None,
                interpolation=False):

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
        self.use_interpolation = interpolation

        if self.path_gen is None:
            self.path_gen = trajectory.circle()
        if self.output_type is None:
            self.output_type = ["rgb"]

        if test_camera_poses is not None:
            self.test_poses = data_utils.load_matrix(test_camera_poses)
            if len(self.test_poses.shape) == 2:
                self.test_poses = self.test_poses.reshape(-1, 4, 4)

            self.test_poses[:, :, 1] = -self.test_poses[:, :, 1]
            self.test_poses[:, :, 2] = -self.test_poses[:, :, 2]
        else:
            self.test_poses = None

    def generate_rays(self, t, intrinsics, img_size, inv_RT=None):
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
        uv = torch.from_numpy(get_uv(h * rh, w * rw, h, w)[0]).type_as(intrinsics)
        # uv = torch.from_numpy(get_uv(h, w, h, w)[0]).type_as(intrinsics)
        uv = uv.reshape(2, -1)
        # intrinsics[0,0] = 1200
        # intrinsics[1,1] = 1200
        # intrinsics[0,2] = 800
        # intrinsics[1,2] = 400

        ray_start = inv_RT[:3, 3]
        ray_dir = geometry.get_ray_direction(ray_start, uv, intrinsics, inv_RT)
        return ray_start[None, :], ray_dir.transpose(0, 1), inv_RT

    @torch.no_grad()    
    def generate(self, models, sample, **kwargs):
        model = models[0]
        model.eval()
        
        # model.field.backbone.points[model.field.backbone.points[:, 1] > 0.5, 1] += model.field.VOXEL_SIZE * 10
        logger.info("rendering starts. {}".format(model.text))

        timer = StopwatchMeter(round=4)
        rgb_path = tempfile.mkdtemp()
        # rgb_path = '/checkpoint/jgu/space/neuralrendering/results/multi_nerf'
        image_names = []
        sample, step = sample

        # fix the rendering size
        a = sample['size'][0,0,0] / self.resolution[0]
        b = sample['size'][0,0,1] / self.resolution[1]
        sample['size'][:, :, 0] /= a
        sample['size'][:, :, 1] /= b
        sample['size'][:, :, 2] *= a
        sample['size'][:, :, 3] *= b

        for shape in range(sample['shape'].size(0)):
            max_step = step + self.frames
            while step < max_step:
                next_step = min(step + self.beam, max_step)
                ray_start, ray_dir, inv_RT = zip(*[
                    self.generate_rays(
                        k, sample['intrinsics'][shape], sample['size'][shape, 0],
                        self.test_poses[k] if self.test_poses is not None else None)
                    for k in range(step, next_step)
                ])
                
                voxels, points = sample.get('voxels', None), sample.get('points', None)
                real_images = sample['full_rgb'] if 'full_rgb' in sample else sample['rgb']
                real_images = real_images.transpose(2, 3) if real_images.size(-1) != 3 else real_images
                _sample = {
                    'id': sample['id'][shape:shape+1],
                    'offset': float((step % self.frames)) / float(self.frames) if self.use_interpolation else None,
                    'rgb': torch.cat([real_images[shape:shape+1] for _ in range(step, next_step)], 1),
                    'ray_start': torch.stack(ray_start, 0).unsqueeze(0),
                    'ray_dir': torch.stack(ray_dir, 0).unsqueeze(0),
                    'extrinsics': torch.stack(inv_RT, 0).unsqueeze(0),
                    'shape': sample['shape'][shape:shape+1],
                    'view': torch.arange(
                        step, next_step, 
                        device=sample['shape'].device).unsqueeze(0),
                    'voxels': voxels[shape:shape+1].clone() if voxels is not None else None,
                    'points': points[shape:shape+1].clone() if points is not None else None,
                    'raymarching_steps': self.raymarching_steps,
                    'size': torch.cat([sample['size'][shape:shape+1] for _ in range(step, next_step)], 1),
                }
                # from fairseq import pdb; pdb.set_trace()
                timer.start()
                outs = model(**_sample)
                timer.stop()

                logger.info("rendering frames: {} {:.4f} {:.4f}".format(step, outs['other_logs']['spf_log'], outs['other_logs']['sp0_log']))

                for k in range(step, next_step):
                    images = model.visualize(
                                _sample, None, 0, k-step, 
                                target_map=True, 
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
                step = next_step

        logger.info("total rendering time: {:.4f}s ({:.4f}s per frame)".format(
            timer.sum, timer.avg / self.beam))
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
            images = [[imageio.imread(file_path) for file_path in output_files if type in file_path] for type in self.output_type]
            images = [np.concatenate([images[j][i] for j, _ in enumerate(self.output_type)], 1) for i in range(len(images[0]))]
            imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, 'full', timestamp), images, fps=self.fps, quality=8)
            # imageio.mimwrite('{}/{}_{}.mp4'.format(self.output_dir, 'full', timestamp), images, fps=self.fps, quality=10)

        # logger.info("cleaning the temple files..")
        # try:
        #     for mydir in set([os.path.dirname(path) for path in output_files]):
        #         shutil.rmtree(mydir)
        # except OSError as e:
        #     print ("Error: %s - %s." % (e.filename, e.strerror))
        #     raise OSError    
        
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
            os.remove(tempfile)
        writer.close()