# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import functools
import cv2
import math
import numpy as np
import imageio
from glob import glob
import os
import copy
import shutil
import skimage
import pandas as pd
import pylab as plt


def load_rgb(path, resolution=None, with_alpha=True, bg_color=-0.8):
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

    # if alpha == 0, make it white (chair)
    img[:, :, :3] = img[:, :, :3] * img[:, :, 3:] + bg_color * (1 - img[:, :, 3:])     
    img = img.transpose(2, 0, 1)
    return img, uv


def load_depth(path, resolution=None, depth_plane=5):
    if path is None:
        return None
    
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


def load_mask(path, resolution=None):
    if path is None:
        return None
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    if resolution is not None:
        h, w = img.shape[:2]
        w, h = resolution, int(h/float(w)*resolution)
        img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
    img = img / img.max()
    return img


def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)


def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        return intrinsics
    except ValueError:
        pass

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
    return full_intrinsic


def unflatten_img(img):
    sizes = img.size()
    side_len = int(math.sqrt(sizes[-1]))
    return img.reshape(*sizes[:-1], side_len, side_len)


def square_crop_img(img):
    if img.shape[0] == img.shape[1]:
        return img  # already square

    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def sample_pixel_from_image(num_pixel, num_sample, mask=None, ratio=1.0, use_bbox=False):
    if mask is None or ratio <= 0.0 or mask.sum() == 0 or (1 - mask).sum() == 0:
        return np.random.choice(num_pixel, num_sample)

    if use_bbox:
        mask = mask.reshape(-1, int(math.sqrt(mask.shape[0])))
        x, y = np.where(mask == 1)
        mask = np.zeros_like(mask)
        mask[x.min(): x.max()+1, y.min(): y.max()+1] = 1.0
        mask = mask.reshape(-1)

    probs = mask * ratio / mask.sum() + (1 - mask) / (num_pixel - mask.sum()) * (1 - ratio)
    # x = np.random.choice(num_pixel, num_sample, True, p=probs)
    return np.random.choice(num_pixel, num_sample, True, p=probs)


def colormap(dz):
    return plt.cm.jet(dz)


def recover_image(img, min_val=-1, max_val=1):
    sizes = img.size()
    side_len = int(sizes[0]**0.5)
    img = ((img - min_val) / (max_val - min_val)).clamp(min=0, max=1).to('cpu')
    if len(sizes) == 1:
        img = torch.from_numpy(colormap(img.numpy()))
    img = img.reshape(side_len, side_len, -1)
    return img

    
def write_images(writer, images, updates): 
    for tag in images:
        img = images[tag]
        tag, dataform = tag.split(':')
        writer.add_image(tag, img, updates, dataformats=dataform)


class InfIndex(object):

    def __init__(self, index_list, shuffle=False):
        self.index_list = index_list
        self.size = len(index_list)
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        if self.shuffle:
            self._perm = np.random.permutation(self.index_list).tolist()
        else:
            self._perm = copy.deepcopy(self.index_list)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return self.size