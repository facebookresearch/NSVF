# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import numpy as np
import torch
import logging

from collections import defaultdict
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
                load_voxel=False, 
                train=True,
                preload=True,
                repeat=1):
        
        if os.path.isdir(paths):
            self.paths = [paths]
        else:
            self.paths = [line.strip() for line in open(paths)]
        
        self.train = train
        self.load_depth = load_depth
        self.load_voxel = load_voxel
        self.max_train_view = max_train_view
        self.num_view = num_view
        self.total_num_shape = len(self.paths)
        self.resolution = resolution
        self.world2camera = True
        self.cache = None
        self.repeat = repeat

        # -- load per-view data
        _data_per_view = defaultdict(lambda: None)
        _data_per_view['rgb'] = self.find_rgb()  
        _data_per_view['ext'] = self.find_extrinsics()
        if self.load_depth:
            _data_per_view['dep'] = self.find_depth()
        _data_per_view['view'] = self.summary_view_data(_data_per_view)

        # -- load per-shape data
        _data_per_shape = defaultdict(lambda: None)
        _data_per_shape['ixt'] = self.find_intrinsics()
        if self.load_voxel:
            _data_per_shape['voxel'] = self.find_voxel()
        _data_per_shape['shape'] = list(range(len(_data_per_shape['ixt'])))

        # group the data..
        data_list = []
        for r in range(repeat):
            # HACK: making several copies to enable multi-GPU usage.
            if r == 0 and preload:
                self.cache = []
                logger.info('pre-load the dataset into memory.')

            for id in range(self.total_num_shape): 
                element = defaultdict(lambda: None)
                total_num_view = len(_data_per_view['rgb'][id])
                perm_ids = np.random.permutation(total_num_view)
                for key in _data_per_view:
                    element[key] = [_data_per_view[key][id][i] for i in perm_ids]
                for key in _data_per_shape:
                    element[key] = _data_per_shape[key][id]
                data_list.append(element)

                if r == 0 and preload:
                    self.cache += [self._load_batch(data_list, id, np.arange(total_num_view))]

        # group the data together
        self.data = data_list
        self.data_index = [
            data_utils.InfIndex(len(d['rgb']), shuffle=self.train) 
        for d in self.data]

    def cutoff(self, file_list):

        def is_empty(list):
            if len(list) == 0:
                raise FileNotFoundError
            return list
        
        # return [files[:self.max_train_view] if train else files[self.max_train_view:] for files in file_list]
        return [is_empty(files[:self.max_train_view]) for files in file_list]

    def find_rgb(self):
        try:
            return self.cutoff([sorted(glob.glob(path + '/rgb/*.png')) for path in self.paths])
        except FileNotFoundError:
            raise FileNotFoundError("CANNOT find rendered images.")
    
    def find_depth(self):
        try:
            return self.cutoff([sorted(glob.glob(path + '/depth/*.exr')) for path in self.paths])
        except FileNotFoundError:
            raise FileNotFoundError("CANNOT find estimateddepths images")

    def find_voxel(self):
        vox_list = []
        for path in self.paths:
            if os.path.exists(path + '/sparse_voxel.txt'):
                vox_list.append(path + '/sparse_voxel.txt')
            else:
                raise FileNotFoundError("CANNOT find intrinsic data")
        return vox_list

    def find_extrinsics(self):
        try:
            return self.cutoff([sorted(glob.glob(path + '/extrinsic/*.txt')) for path in self.paths])
        except FileNotFoundError:
            try:
                self.world2camera = False
                return self.cutoff([sorted(glob.glob(path + '/pose/*.txt')) for path in self.paths])
            except FileNotFoundError:
                raise FileNotFoundError('world2camera or camera2world matrices not found.')   
    
    def find_intrinsics(self):
        ixt_list = []
        for path in self.paths:
            if os.path.exists(path + '/intrinsic.txt'):
                ixt_list.append(path + '/intrinsic.txt')
            else:
                raise FileNotFoundError("CANNOT find intrinsic data")
        return ixt_list

    def summary_view_data(self, _data_per_view):
        keys = [*_data_per_view]
        num_of_objects = len(_data_per_view[keys[0]])
        for k in range(num_of_objects):
            assert len(set([len(_data_per_view[key][k]) for key in keys])) == 1, "numer of views must be consistent."
        return [list(range(len(_data_per_view[keys[0]][k]))) for k in range(num_of_objects)]
    
    def ordered_indices(self):
        return np.arange(len(self.data))

    def num_tokens(self, index):
        return self.num_view * self.resolution ** 2  

    def _load_view(self, packed_data, view_idx):
        image, uv = data_utils.load_rgb(packed_data['rgb'][view_idx], resolution=self.resolution)
        rgb, alpha = image[:3], image[3]  # C x H x W for RGB
        extrinsics = data_utils.load_matrix(packed_data['ext'][view_idx])
        extrinsics = geometry.parse_extrinsics(extrinsics, self.world2camera)
        z = data_utils.load_depth(packed_data['dep'][view_idx], resolution=self.resolution) \
            if packed_data['dep'] is not None else None

        return {
            'view': view_idx,
            'uv': uv.reshape(2, -1), 
            'rgb': rgb.reshape(3, -1), 
            'alpha': alpha.reshape(-1), 
            'extrinsics': extrinsics,
            'depths': z.reshape(-1) if z is not None else None
        }
    
    def _load_shape(self, packed_data):        
        intrinsics = data_utils.load_intrinsics(packed_data['ixt'])[0]
        return {'intrinsics': intrinsics}

    def _load_batch(self, data, index, view_ids=None):
        if view_ids is None:
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
        # from fairseq import pdb; pdb.set_trace()
        return index, self._load_shape(data[index]), [self._load_view(data[index], view_id) for view_id in view_ids]

    def __getitem__(self, index):
        if self.cache is not None:
            index = index // self.repeat
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
            return self.cache[index][0], self.cache[index][1], [self.cache[index][2][i] for i in view_ids]
        return self._load_batch(self.data, index)

    def collater(self, samples):
        results = {}
        
        results['shape'] = torch.from_numpy(np.array([s[0] for s in samples]))
        for key in samples[0][1]:
            results[key] = torch.from_numpy(
                np.array([s[1][key] for s in samples])
            ) if samples[0][1][key] is not None else None
        for key in samples[0][2][0]:
            results[key] = torch.from_numpy(
                np.array([[d[key] for d in s[2]] for s in samples])
            ) if samples[0][2][0][key] is not None else None
        
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
        index, data_per_shape, data_per_view = self.dataset[index]

        # sample pixels from the original images
        sample_index = [
            data_utils.sample_pixel_from_image(
                data['alpha'], self.num_sample, 
                ignore_mask=self.ignore_mask)
            for data in data_per_view
        ]

        for i, data in enumerate(data_per_view):
            data_per_view[i]['full_rgb'] = data['rgb']
            for key in data:
                if data[key] is not None \
                    and len(data[key].shape) == 2 \
                    and data[key].shape[1] > self.num_sample \
                    and key != 'full_rgb':

                    data_per_view[i][key] = data[key][:, sample_index[i]] 
        
        return index, data_per_shape, data_per_view

    def num_tokens(self, index):
        return self.dataset.num_view * self.num_sample 

class WorldCoordDataset(BaseWrapperDataset):
    """
    A wrapper dataset. transform UV space into World space
    """
    def __getitem__(self, index):
        index, data_per_shape, data_per_view = self.dataset[index]

        def camera2world(data):
            inv_RT = data['extrinsics']
            intrinsics = data_per_shape['intrinsics']

            # get camera center (XYZ)
            ray_start = inv_RT[:3, 3]
            
            # get points at a random depth (=1)
            # OR transform depths to world coordinates (optional)
            if data.get('depths', None) is None:
                rt_cam = geometry.uv2cam(data['uv'], 1, intrinsics, True)
            else:
                rt_cam = geometry.uv2cam(data['uv'], data['depths'], intrinsics, True)
                
            rt = geometry.cam2world(rt_cam, inv_RT)
            
            # get the ray direction
            # ray_dir, depth = geometry.normalize(rt - ray_start[:, None], axis=0)
            ray_dir, _ = geometry.normalize(rt - ray_start[:, None], axis=0)

            # here we still keep the original data for tracking purpose
            data.update({
                'ray_start': ray_start,
                'ray_dir': ray_dir,
            })
            return data

        return index, data_per_shape, [camera2world(data) for data in data_per_view]
        
    def collater(self, samples):
        results = self.dataset.collater(samples)
        results['ray_dir'] = results['ray_dir'].transpose(2, 3)
        results['rgb'] = results['rgb'].transpose(2, 3)
        return results
