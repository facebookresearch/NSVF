# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import numpy as np
import torch

from fairseq.data import FairseqDataset, BaseWrapperDataset
from . import data_utils, geometry

# from .collaters import Seq2SeqCollater


class RenderedImageDataset(FairseqDataset):
    """
    A dataset contains a series of images renderred offline for an object.
    """

    def __init__(self, paths, max_train_view, num_view, resolution=None, train=True):
        if os.path.isdir(paths):
            self.paths = [paths]
        else:
            self.paths = [line.strip() for line in open(paths)]
        
        self.train = train
        self.max_train_view = max_train_view
        self.num_view = num_view
        self.resolution = resolution

        # fetch the image paths
        def cutoff(file_list):
            return [files[:self.max_train_view] if train else files[self.max_train_view:] for files in file_list]

        rgb_list = cutoff([sorted(glob.glob(path + '/rgb/model_*.png')) for path in self.paths])
        ext_list = cutoff([sorted(glob.glob(path + '/extrinsic/model_*.txt')) for path in self.paths])
        ixt_list = [path + '/intrinsic.txt' for path in self.paths]
        
        # group the data
        _data_list = [(list(zip(r, e)), i) for r, e, i in zip(rgb_list, ext_list, ixt_list)]
        
        # HACK: trying to save several duplicates for multi-GPU usage.
        data_list = []
        for f, i in _data_list:
            data_list += [(np.random.permutation(f).tolist(), i) for _ in range(max_train_view // num_view)]

        # group the data together
        self.data = data_list
        self.data_index = [data_utils.InfIndex(len(d[0]), shuffle=True) for d in self.data]

    def ordered_indices(self):
        return np.arange(len(self.data))

    def num_tokens(self, index):
        return len(self.data[index][1])

    def __getitem__(self, index):
    
        def load_data(packed_data, img_idx):
            image, uv = data_utils.load_rgb(packed_data[img_idx][0], resolution=self.resolution)
            extrinsics = data_utils.load_matrix(packed_data[img_idx][1])
            rgb, alpha = image[:3], image[3]  # C x H x W for RGB
            return {
                'uv': uv.reshape(2, -1), 
                'rgb': rgb.reshape(3, -1), 
                'alpha': alpha.reshape(-1), 
                'extrinsics': extrinsics
            }

        return [
            load_data(self.data[index][0], next(self.data_index[index])) 
            for _ in range(self.num_view)
        ], data_utils.load_matrix(self.data[index][1])

    def collater(self, samples):
        uv =  np.array([[data['uv'] for data in sample[0]] for sample in samples])
        rgb = np.array([[data['rgb'] for data in sample[0]] for sample in samples])
        alpha = np.array([[data['alpha'] for data in sample[0]] for sample in samples])
        extrinsics = np.array([[data['extrinsics'] for data in sample[0]] for sample in samples])
        intrinsics = np.array([sample[1] for sample in samples])

        # from fairseq import pdb; pdb.set_trace()
        # transform to tensor
        return {
            'uv': torch.from_numpy(uv),                 # BV2(HW)
            'rgb': torch.from_numpy(rgb),               # BV3(HW)
            'alpha': torch.from_numpy(alpha),           # BV(HW)
            'extrinsic': torch.from_numpy(extrinsics),  # BV34
            'intrinsic': torch.from_numpy(intrinsics)   # B33
        }


class SampledPixelDataset(BaseWrapperDataset):
    """
    A wrapper dataset, which split rendered images into pixels
    """

    def __init__(self, dataset, num_sample=None):
        super().__init__(dataset)
        self.num_sample = num_sample

    def __getitem__(self, index):
        packed_data, intrinsics = self.dataset[index]

        # sample pixels from the original images
        sample_index = [
            data_utils.sample_pixel_from_image(
                data['alpha'], self.num_sample)
            for data in packed_data
        ]
        packed_data = [
            {
                'uv': data['uv'][:, sample_index[i]],
                'rgb': data['rgb'][:, sample_index[i]],
                'alpha': data['alpha'][sample_index[i]],
                'extrinsics': data['extrinsics']
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
            RT = np.vstack([data['extrinsics'], np.array([[0, 0, 0, 1.0]])])
            inv_RT = np.linalg.inv(RT)

            # get camera center (XYZ)
            ray_start = inv_RT[:3, 3]

            # get points at a random depth (=1)
            rt_cam = geometry.uv2cam(data['uv'], 1, intrinsics, True)
            rt = geometry.cam2world(rt_cam, inv_RT)
            
            # get the ray direction
            ray_dir = geometry.normalize(rt - ray_start[:, None], axis=0)
            
            return {'ray_start': ray_start, 'ray_dir': ray_dir, 'rgb': data['rgb'], 'alpha': data['alpha']}

        return [camera2world(data) for data in packed_data]
        
    def collater(self, samples):
        ray_start =  np.array([[data['ray_start'] for data in sample] for sample in samples])
        ray_dir = np.array([[data['ray_dir'] for data in sample] for sample in samples])
        rgb = np.array([[data['rgb'] for data in sample] for sample in samples])
        alpha = np.array([[data['alpha'] for data in sample] for sample in samples])

        # transform to tensor
        return {
            'ray_start': torch.from_numpy(ray_start),                   # BV3
            'ray_dir': torch.from_numpy(ray_dir).transpose(2, 3),     # BVN3
            'rgb': torch.from_numpy(rgb).transpose(2, 3), # BVN3
            'alpha': torch.from_numpy(alpha),             # BVN
        }
