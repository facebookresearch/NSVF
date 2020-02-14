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
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
        uv = uv[:, ::img_size//resolution, ::img_size//resolution]

    img[:, :, :3] -= 0.5
    img[:, :, :3] *= 2.
    img = img.transpose(2, 0, 1)
    return img, uv


def load_depth(path, resolution=None, depth_plane=5):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)

    ret,img = cv2.threshold(img, depth_plane, depth_plane, cv2.THRESH_TRUNC)
    if resolution is not None:
        h, w = img.shape[:2]
        w, h = resolution, int(h/float(w)*resolution)
        img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        #img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    #img *= 1e-4
    if len(img.shape) ==3:
        img = img[:,:,:1]
        img = img.transpose(2,0,1)
    else:
        img = img[None,:,:]
    return img


def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)


def load_intrinsics(filepath, resized_width=None, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        scale = float(file.readline())
        #print(file.readline())
        file.readline()#what's this ? skip
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)
    
    if resized_width is not None:
        resized_height = int(height/float(width)*resized_width)

        cx = cx/width * resized_width
        cy = cy/height * resized_height
        f = resized_width / width * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    return full_intrinsic, grid_barycenter, scale, world2cam_poses


def square_crop_img(img):
    if img.shape[0] == img.shape[1]:
        return img  # already square

    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def sample_pixel_from_image(alpha, num_sample, ignore_mask=1.0):
    noise = np.random.uniform(size=alpha.shape)
    # if ignore_mask < 1:
    #     _mask = alpha.astype(np.float32)
    #     _mask = _mask + (1 - _mask) * ignore_mask
    #     noise = noise * _mask
    noise = noise.reshape(-1)
    return np.argsort(noise)[-num_sample:]


def write_images(writer, images): 
    from fairseq import metrics
    for tag in images:
        img = images[tag]
        tag, dataform = tag.split(':')
        writer.add_image(tag, img, 
            metrics.get_meter('default', 'num_updates').val,
            dataformats=dataform)


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