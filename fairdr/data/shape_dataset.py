# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import copy
import numpy as np
import torch
import logging

from collections import defaultdict
from fairseq.data import FairseqDataset, BaseWrapperDataset
from . import data_utils, geometry, trajectory


logger = logging.getLogger(__name__)


class ShapeDataset(FairseqDataset):
    """
    A dataset that only returns data per shape
    """
    def __init__(self, 
                paths, 
                load_point=False,
                preload=True,
                repeat=1):
        
        if os.path.isdir(paths):
            self.paths = [paths]
        else:
            self.paths = [line.strip() for line in open(paths)]

        self.load_point = load_point
        self.total_num_shape = len(self.paths)
        self.cache = None
        self.repeat = repeat

        # -- load per-shape data
        _data_per_shape = {}
        _data_per_shape['ixt'] = self.find_intrinsics()
        if self.load_point:
            _data_per_shape['pts'] = self.find_point()
        _data_per_shape['shape'] = list(range(len(_data_per_shape['ixt'])))

        # group the data..
        data_list = []
        for r in range(repeat):
            # HACK: making several copies to enable multi-GPU usage.
            if r == 0 and preload:
                self.cache = []
                logger.info('pre-load the dataset into memory.')

            for id in range(self.total_num_shape): 
                element = {}
                for key in _data_per_shape:
                    element[key] = _data_per_shape[key][id]
                data_list.append(element)

                if r == 0 and preload:
                    self.cache += [self._load_batch(data_list, id)]

        # group the data together
        self.data = data_list

    def find_point(self):
        vox_list = []
        for path in self.paths:
            if os.path.exists(path + '/sparse_voxel.txt'):
                vox_list.append(path + '/sparse_voxel.txt')
            else:
                raise FileNotFoundError("CANNOT find intrinsic data")
        return vox_list

    def find_intrinsics(self):
        ixt_list = []
        for path in self.paths:
            if os.path.exists(path + '/intrinsic.txt'):
                ixt_list.append(path + '/intrinsic.txt')
            else:
                raise FileNotFoundError("CANNOT find intrinsic data")
        return ixt_list

    def _load_shape(self, packed_data):        
        intrinsics = data_utils.load_intrinsics(packed_data['ixt']).astype('float32')
        if packed_data.get('pts', None) is not None:
            voxels = data_utils.load_matrix(packed_data['pts'])
            voxels, points = voxels[:, :3].astype('int32'), voxels[:, 3:]
        else:
            voxels, points = None, None

        return {'intrinsics': intrinsics, 'voxels': voxels, 'points': points}

    def _load_batch(self, data, index):
        return index, self._load_shape(data[index])

    def __getitem__(self, index):
        if self.cache is not None:
            return self.cache[index % self.total_num_shape][0], \
                   self.cache[index % self.total_num_shape][1]
        return self._load_batch(self.data, index)

    def ordered_indices(self):
        return np.arange(len(self.data))

    def num_tokens(self, index):
        return 1

    def collater(self, samples):
        results = {}
        
        results['shape'] = torch.from_numpy(np.array([s[0] for s in samples]))    
        for key in samples[0][1]:
            if samples[0][1][key] is not None:
                if key == 'voxels':
                    # save for sparse-conv
                    batch_voxels = np.concatenate(
                        [
                            np.concatenate(
                                [s[1][key], j * np.ones((s[1][key].shape[0], 1))], 1) 
                            for j, s in enumerate(samples)
                        ],
                        axis=0
                    )
                    results[key] = torch.from_numpy(batch_voxels).int()
                
                elif key == "points":
                    # save for pointnet++, handling dynamic number of points by duplicating points.
                    max_num = max(s[1][key].shape[0] for s in samples)
                    batch_points = np.array([
                        np.concatenate(
                            [s[1][key], s[1][key][:(max_num - s[1][key].shape[0])]], 0)
                            if s[1][key].shape[0] < max_num
                            else s[1][key]
                        for s in samples])
                    results[key] = torch.from_numpy(batch_points)
                
                else:
                    results[key] = torch.from_numpy(
                        np.array([s[1][key] for s in samples]))
            else:
                results[key] = None

        return results


class ShapeViewDataset(ShapeDataset):
    """
    A dataset contains a series of images renderred offline for an object.
    """

    def __init__(self, 
                paths, 
                max_train_view, 
                num_view, 
                resolution=None, 
                load_depth=False,
                load_mask=False,
                load_point=False,
                train=True,
                preload=True,
                repeat=1):
        
        super().__init__(paths, load_point, False, repeat)

        self.train = train
        self.load_depth = load_depth
        self.load_mask = load_mask
        self.max_train_view = max_train_view
        self.num_view = num_view
        self.resolution = resolution
        self.world2camera = True
        self.cache_view = None

        # -- load per-view data
        _data_per_view = {}
        _data_per_view['rgb'] = self.find_rgb()  
        _data_per_view['ext'] = self.find_extrinsics()

        if self.load_depth:
            _data_per_view['dep'] = self.find_depth()
        if self.load_mask:
            _data_per_view['mask'] = self.find_mask()
        
        _data_per_view['view'] = self.summary_view_data(_data_per_view)


        # group the data.
        _index = 0
        for r in range(repeat):
            # HACK: making several copies to enable multi-GPU usage.
            if r == 0 and preload:
                self.cache = []
                logger.info('pre-load the dataset into memory.')

            for id in range(self.total_num_shape): 
                element = {}
                total_num_view = len(_data_per_view['rgb'][id])
                perm_ids = np.random.permutation(total_num_view)
                for key in _data_per_view:
                    element[key] = [_data_per_view[key][id][i] for i in perm_ids]
                self.data[_index].update(element)

                if r == 0 and preload:
                    self.cache += [self._load_batch(self.data, id, np.arange(total_num_view))]
                _index += 1

        # group the data together
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
            raise FileNotFoundError("CANNOT find estimated depths images") 

    def find_mask(self):
        try:
            return self.cutoff([sorted(glob.glob(path + '/mask/*.png')) for path in self.paths])
        except FileNotFoundError:
            raise FileNotFoundError("CANNOT find precomputed mask images")

    def find_extrinsics(self):
        try:
            return self.cutoff([sorted(glob.glob(path + '/extrinsic/*.txt')) for path in self.paths])
        except FileNotFoundError:
            try:
                self.world2camera = False
                return self.cutoff([sorted(glob.glob(path + '/pose/*.txt')) for path in self.paths])
            except FileNotFoundError:
                raise FileNotFoundError('world2camera or camera2world matrices not found.')   

    def summary_view_data(self, _data_per_view):
        keys = [*_data_per_view]
        num_of_objects = len(_data_per_view[keys[0]])
        for k in range(num_of_objects):
            assert len(set([len(_data_per_view[key][k]) for key in keys])) == 1, "numer of views must be consistent."
        return [list(range(len(_data_per_view[keys[0]][k]))) for k in range(num_of_objects)]

    def num_tokens(self, index):
        return self.num_view * self.resolution ** 2  

    def _load_view(self, packed_data, view_idx):
        image, uv = data_utils.load_rgb(packed_data['rgb'][view_idx], resolution=self.resolution)
        rgb, alpha = image[:3], image[3]  # C x H x W for RGB
        extrinsics = data_utils.load_matrix(packed_data['ext'][view_idx])
        extrinsics = geometry.parse_extrinsics(extrinsics, self.world2camera).astype('float32')
        z, mask = None, None
        if packed_data.get('dep', None) is not None:
            z = data_utils.load_depth(packed_data['dep'][view_idx], resolution=self.resolution)
        if packed_data.get('mask', None) is not None:
            mask = data_utils.load_mask(packed_data['mask'][view_idx], resolution=self.resolution)

        return {
            'view': view_idx,
            'uv': uv.reshape(2, -1), 
            'rgb': rgb.reshape(3, -1), 
            'alpha': alpha.reshape(-1), 
            'extrinsics': extrinsics,
            'depths': z.reshape(-1) if z is not None else None,
            'mask': mask.reshape(-1) if mask is not None else None,
        }

    def _load_batch(self, data, index, view_ids=None):
        if view_ids is None:
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
        return index, self._load_shape(data[index]), [self._load_view(data[index], view_id) for view_id in view_ids]

    def __getitem__(self, index):
        if self.cache is not None:
            view_ids = [next(self.data_index[index]) for _ in range(self.num_view)]
            return copy.deepcopy(self.cache[index % self.total_num_shape][0]), \
                   copy.deepcopy(self.cache[index % self.total_num_shape][1]), \
                  [copy.deepcopy(self.cache[index % self.total_num_shape][2][i]) for i in view_ids]
        return self._load_batch(self.data, index)

    def collater(self, samples):
        results = super().collater(samples)
        for key in samples[0][2][0]:
            results[key] = torch.from_numpy(
                np.array([[d[key] for d in s[2]] for s in samples])
            ) if samples[0][2][0].get(key, None) is not None else None
        return results


class SampledPixelDataset(BaseWrapperDataset):
    """
    A wrapper dataset, which split rendered images into pixels
    """

    def __init__(self, dataset, num_sample=None, sampling_on_mask=1.0, sampling_on_bbox=False):
        super().__init__(dataset)
        self.num_sample = num_sample
        self.sampling_on_mask = sampling_on_mask
        self.sampling_on_bbox = sampling_on_bbox

    def __getitem__(self, index):
        index, data_per_shape, data_per_view = self.dataset[index]
        
        # sample pixels from the original images
        sample_index = [
            data_utils.sample_pixel_from_image(
                data['alpha'].shape[-1], 
                self.num_sample, 
                data.get('mask', None),
                self.sampling_on_mask,
                self.sampling_on_bbox)
            for data in data_per_view
        ]

        for i, data in enumerate(data_per_view):
            data_per_view[i]['full_rgb'] = copy.deepcopy(data['rgb'])
            for key in data:
                if data[key] is not None \
                    and (key != 'extrinsics' and key != 'view' and key != 'full_rgb') \
                    and data[key].shape[-1] > self.num_sample:

                    if len(data[key].shape) == 2:
                        data_per_view[i][key] = data[key][:, sample_index[i]] 
                    else:
                        data_per_view[i][key] = data[key][sample_index[i]]
            data_per_view[i]['index'] = sample_index[i]
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
            ray_dir = geometry.get_ray_direction(
                ray_start, data['uv'], intrinsics, inv_RT, 1
            )
            
            # here we still keep the original data for tracking purpose
            data.update({
                'ray_start': ray_start,
                'ray_dir': ray_dir,
            })
            return data

        return index, data_per_shape, [camera2world(data) for data in data_per_view]
        
    def collater(self, samples):
        results = self.dataset.collater(samples)
        results['ray_start'] = results['ray_start'].unsqueeze(-2)
        results['ray_dir'] = results['ray_dir'].transpose(2, 3)
        results['rgb'] = results['rgb'].transpose(2, 3)
        if results.get('full_rgb', None) is not None:
            results['full_rgb'] = results['full_rgb'].transpose(2, 3)
        return results

