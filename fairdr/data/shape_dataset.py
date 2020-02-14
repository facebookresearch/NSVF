# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import numpy as np
import torch
import logging

from fairseq.data import FairseqDataset, BaseWrapperDataset
from . import data_utils, geometry


logger = logging.getLogger(__name__)


class RenderedImageDataset(FairseqDataset):
    """
    A dataset contains a series of images renderred offline for an object.
    """

    def __init__(self, 
                paths, 
                max_train_view, 
                num_view, 
                resolution=None, 
                load_depth=False, 
                train=True,
                preload=True,
                repeat=1):
        
        if os.path.isdir(paths):
            self.paths = [paths]
        else:
            self.paths = [line.strip() for line in open(paths)]
        
        self.train = train
        self.load_depth = load_depth
        self.max_train_view = max_train_view
        self.num_view = num_view
        self.resolution = resolution
        self.world2camera = True

        # fetch the image paths
        def cutoff(file_list):
            # return [files[:self.max_train_view] if train else files[self.max_train_view:] for files in file_list]
            return [files[:self.max_train_view] for files in file_list]

        # get datasets
        ixt_list = [path + '/intrinsic.txt' for path in self.paths]
        rgb_list = cutoff([sorted(glob.glob(path + '/rgb/*.png')) for path in self.paths])
        ext_list = cutoff([sorted(glob.glob(path + '/extrinsic/*.txt')) for path in self.paths])
        if len(ext_list[0]) == 0:
            ext_list = cutoff([sorted(glob.glob(path + '/pose/*.txt')) for path in self.paths])
            self.world2camera = False
            if len(ext_list[0]) == 0:
                raise FileNotFoundError('world2camera or camera2world matrices not found.')
        
        # load depth
        if self.load_depth:
            dep_list = cutoff([sorted(glob.glob(path + '/depth/*.exr')) for path in self.paths])
            if len(dep_list[0]) == 0:
                raise FileNotFoundError('depth map does not found.')
        
        # group the data
        _data_list = []
        
        for id in range(len(ixt_list)):
            _data_element = [rgb_list[id], ext_list[id]]
            if self.load_depth:
                _data_element.append(dep_list[id])
            _data_list.append((list(zip(*_data_element)), ixt_list[id]))
        
        # HACK: trying to save several duplicates for multi-GPU usage.
        data_list = []
        for f, i in _data_list:
            for _ in range(repeat):
                data_list += [(np.random.permutation(f).tolist(), i)]
                
        # group the data together
        self.data = data_list
        self.data_index = [data_utils.InfIndex(len(d[0]), shuffle=True) for d in self.data]

        if preload:  # read everything into memory
            self.cache = [self.load_batch(i) for i in range(len(self.data))]
            logger.info('pre-load the dataset into memory.')
        else:
            self.cache = None

    def ordered_indices(self):
        return np.arange(len(self.data))

    def num_tokens(self, index):
        return len(self.data[index][1])

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache[index]
        return self.load_batch(index)

    def load_batch(self, index):

        def _load_data(packed_data, img_idx):
            image, uv = data_utils.load_rgb(packed_data[img_idx][0], resolution=self.resolution)
            rgb, alpha = image[:3], image[3]  # C x H x W for RGB
            extrinsics = data_utils.load_matrix(packed_data[img_idx][1])
            extrinsics = geometry.parse_extrinsics(extrinsics, self.world2camera)
            
            if self.load_depth:
                z = data_utils.load_depth(packed_data[img_idx][2], resolution=self.resolution)
            else:
                z = None

            return {
                'shape': index,
                'view': img_idx,
                'uv': uv.reshape(2, -1), 
                'rgb': rgb.reshape(3, -1), 
                'alpha': alpha.reshape(-1), 
                'extrinsics': extrinsics,
                'z': z.reshape(-1) if z is not None else None
            }

        
        try:
            intrinsics = data_utils.load_matrix(self.data[index][1])
        except ValueError:
            intrinsics = data_utils.load_intrinsics(self.data[index][1])[0]

        return [
            _load_data(self.data[index][0], next(self.data_index[index])) 
            for _ in range(self.num_view)
        ], intrinsics

    def collater(self, samples):
        results = {
            key: torch.from_numpy(
                np.array([[data[key] for data in sample[0]] for sample in samples]))
                if samples[0][0][key] is not None else None
            for key in ('shape', 'view', 'uv', 'rgb', 'alpha', 'extrinsics', 'z')
        }
        results['intrinsics'] = torch.from_numpy(
            np.array([sample[1] for sample in samples])) 
        return results


class SampledPixelDataset(BaseWrapperDataset):
    """
    A wrapper dataset, which split rendered images into pixels
    """

    def __init__(self, dataset, num_sample=None):
        super().__init__(dataset)
        self.num_sample = num_sample
        self.ignore_mask = 1.1

    def __getitem__(self, index):
        packed_data, intrinsics = self.dataset[index]

        # sample pixels from the original images
        sample_index = [
            data_utils.sample_pixel_from_image(
                data['alpha'], self.num_sample, 
                ignore_mask=self.ignore_mask)
            for data in packed_data
        ]

        packed_data = [
            {
                'shape': data['shape'],
                'view': data['view'],
                'uv': data['uv'][:, sample_index[i]],
                'rgb': data['rgb'][:, sample_index[i]],
                'alpha': data['alpha'][sample_index[i]],
                'extrinsics': data['extrinsics'],
                'z': data['z'][sample_index[i]] 
                    if data['z'] is not None else None
            }
            for i, data in enumerate(packed_data)
        ]
        return packed_data, intrinsics


class WorldCoordDataset(BaseWrapperDataset):
    """
    A wrapper dataset. transform UV space into World space
    """
    def __getitem__(self, index):
        packed_data, intrinsics = self.dataset[index]

        def camera2world(data):
            inv_RT = data['extrinsics']

            # get camera center (XYZ)
            ray_start = inv_RT[:3, 3]
            
            # get points at a random depth (=1)
            # OR transform depths to world coordinates (optional)
            if data.get('z', None) is None:
                rt_cam = geometry.uv2cam(data['uv'], 1, intrinsics, True)
            else:
                rt_cam = geometry.uv2cam(data['uv'], data['z'], intrinsics, True)
                
            rt = geometry.cam2world(rt_cam, inv_RT)
            
            # get the ray direction
            ray_dir, depth = geometry.normalize(rt - ray_start[:, None], axis=0)
        
            return {
                'shape': data['shape'],
                'view': data['view'],
                'ray_start': ray_start, 
                'ray_dir': ray_dir, 
                'rgb': data['rgb'], 
                'alpha': data['alpha'],
                'depths': depth if data['z'] is not None else None
            }

        return [camera2world(data) for data in packed_data]
        
    def collater(self, samples):
        results = {
            key: torch.from_numpy(
                np.array([[data[key] for data in sample] for sample in samples]))
                if samples[0][0][key] is not None else None
            for key in ('shape', 'view', 'alpha', 'rgb', 'ray_start', 'ray_dir', 'depths')
        }
        results['ray_dir'] = results['ray_dir'].transpose(2, 3)
        results['rgb'] = results['rgb'].transpose(2, 3)

        return results
