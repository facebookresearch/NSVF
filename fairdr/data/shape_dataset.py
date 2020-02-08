# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, glob
import numpy as np

from fairseq.data import FairseqDataset, BaseWrapperDataset
from . import data_utils

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
    
        def load_data(data, img_idx):
            packed_data, ixt = data
            image = data_utils.load_rgb(packed_data[img_idx][0], resolution=self.resolution)
            pose = data_utils.load_matrix(packed_data[img_idx][1])
            rgb, alpha = image[:3], image[3]  # C x H x W for RGB

            return {'rgb': rgb, 'alpha': alpha, 'pose': pose}

        return [
            load_data(self.data[index], next(self.data_index[index])) 
            for _ in range(self.num_view)
        ]

    def collater(self, samples):
        
        # gather
        rgb = np.array([[data['rgb'] for data in sample] for sample in samples])
        alpha = np.array([[data['alpha'] for data in sample] for sample in samples])
        pose = np.array([[data['pose'] for data in sample] for sample in samples])
        
        from fairseq import pdb; pdb.set_trace()
        # transform to tensor
        return {
            'rgb': torch.from_numpy(rgb),      # BVCHW
            'alpha': torch.from_numpy(alpha),  # BVHW
            'pose': torch.from_numpy(pose)     # BV34
        }


class SampledPixelDataset(BaseWrapperDataset):
    """
    A wrapper dataset, which split rendered images into pixels
    """

    def __init__(self, dataset, num_sample=None):
        super().__init__(dataset)

        self.num_sample = num_sample
