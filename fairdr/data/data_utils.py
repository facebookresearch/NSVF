# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import functools
import cv2
import numpy as np
import imageio
from glob import glob
import os
import shutil
import skimage
import pandas as pd


def load_rgb(path, resolution=None, with_alpha=True):
    if with_alpha:
        img = imageio.imread(path)  # RGB-ALPHA
    else:
        img = imageio.imread(path)[:, :, :3]

    img = skimage.img_as_float32(img)
    img = square_crop_img(img)
    img_size = img.shape[0]

    # uv coordinates
    uv = np.flip(np.mgrid[0: img_size, 0: img_size], axis=0).astype(np.float32)

    if resolution is not None:
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_AREA)
        uv = uv[:, ::img_size//resolution, ::img_size//resolution]

    img[:, :, :3] -= 0.5
    img[:, :, :3] *= 2.
    img = img.transpose(2, 0, 1)
    return img, uv


def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)])


def square_crop_img(img):
    if img.shape[0] == img.shape[1]:
        return img  # already square

    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def sample_pixel_from_image(alpha, num_sample):
    noise = np.random.uniform(size=alpha.shape)
    noise = noise.reshape(-1)
    return np.argsort(noise)[:num_sample]


class InfIndex(object):

    def __init__(self, size, shuffle=False):
        self.size = size
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = self.size
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return self.size