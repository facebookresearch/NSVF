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
import skimage.metrics
import pandas as pd
import pylab as plt
import fairseq.distributed_utils as du

from plyfile import PlyData, PlyElement
from fairseq.meters import StopwatchMeter

def get_rank():    
    try:
        return du.get_rank()
    except AssertionError:
        return 0


def get_world_size():
    try:
        return du.get_world_size()
    except AssertionError:
        return 1
        

def parse_views(view_args):
    output = []
    try:
        xx = view_args.split(':')
        ids = xx[0].split(',')
        for id in ids:
            if '..' in id:
                a, b = id.split('..')
                output += list(range(int(a), int(b)))
            else:
                output += [int(id)]
        if len(xx) > 1:
            output = output[::int(xx[-1])]
    except Exception as e:
        raise Exception("parse view args error: {}".format(e))

    return output


def get_uv(H, W, h, w):
    """
    H, W: real image (intrinsics)
    h, w: resized image
    """
    uv = np.flip(np.mgrid[0: h, 0: w], axis=0).astype(np.float32)
    uv[0] = uv[0] * float(W / w)
    uv[1] = uv[1] * float(H / h)
    return uv, [float(H / h), float(W / w)]


def load_rgb(
    path, 
    resolution=None, 
    with_alpha=True, 
    bg_color=[1.0, 1.0, 1.0],
    min_rgb=-1,
    interpolation='AREA'):
    if with_alpha:
        img = imageio.imread(path)  # RGB-ALPHA
    else:
        img = imageio.imread(path)[:, :, :3]

    img = skimage.img_as_float32(img).astype('float32')
    H, W, D = img.shape
    h, w = resolution
    
    if D == 3:
        img = np.concatenate([img, np.ones((img.shape[0], img.shape[1], 1))], -1).astype('float32')
    
    uv, ratio = get_uv(H, W, h, w)
    if (h < H) or (w < W):
        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA).astype('float32')

    if min_rgb == -1:  # 0, 1  --> -1, 1
        img[:, :, :3] -= 0.5
        img[:, :, :3] *= 2.

    img[:, :, :3] = img[:, :, :3] * img[:, :, 3:] + np.asarray(bg_color)[None, None, :] * (1 - img[:, :, 3:])
    img[:, :, 3] = img[:, :, 3] * (img[:, :, :3] != np.asarray(bg_color)[None, None, :]).any(-1) 
    img = img.transpose(2, 0, 1)
    
    return img, uv, ratio


def load_depth(path, resolution=None, depth_plane=5):
    if path is None:
        return None
    
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    # ret, img = cv2.threshold(img, depth_plane, depth_plane, cv2.THRESH_TRUNC)
    
    H, W = img.shape[:2]
    h, w = resolution
    if (h < H) or (w < W):
        img  = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
        #img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

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
    h, w = resolution
    H, W = img.shape[:2]
    if (h < H) or (w < W):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST).astype('float32')
    img = img / (img.max() + 1e-7)
    return img


def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)


def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:
        
        f, cx, cy, _ = map(float, file.readline().split())
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


def unflatten_img(img, width=512):
    sizes = img.size()
    height = sizes[-1] // width
    return img.reshape(*sizes[:-1], height, width)


def square_crop_img(img):
    if img.shape[0] == img.shape[1]:
        return img  # already square

    min_dim = np.amin(img.shape[:2])
    center_coord = np.array(img.shape[:2]) // 2
    img = img[center_coord[0] - min_dim // 2:center_coord[0] + min_dim // 2,
          center_coord[1] - min_dim // 2:center_coord[1] + min_dim // 2]
    return img


def sample_pixel_from_image(
    num_pixel, num_sample, 
    mask=None, ratio=1.0,
    use_bbox=False, 
    center_ratio=1.0,
    width=512,
    patch_size=1):
    
    if patch_size > 1:
        assert (num_pixel % (patch_size * patch_size) == 0) \
            and (num_sample % (patch_size * patch_size) == 0), "size must match"
        _num_pixel = num_pixel // (patch_size * patch_size)
        _num_sample = num_sample // (patch_size * patch_size)
        height = num_pixel // width

        _mask = None if mask is None else \
            mask.reshape(height, width).reshape(
                height//patch_size, patch_size, width//patch_size, patch_size
            ).any(1).any(-1).reshape(-1)
        _width = width // patch_size
        _out = sample_pixel_from_image(_num_pixel, _num_sample, _mask, ratio, use_bbox, _width)
        _x, _y = _out % _width, _out // _width
        x, y = _x * patch_size, _y * patch_size
        x = x[:, None, None] + np.arange(patch_size)[None, :, None] 
        y = y[:, None, None] + np.arange(patch_size)[None, None, :]
        out = x + y * width
        return out.reshape(-1)

    if center_ratio < 1.0:
        r = (1 - center_ratio) / 2.0
        H, W = num_pixel // width, width
        mask0 = np.zeros((H, W))
        mask0[int(H * r): H - int(H * r), int(W * r): W - int(W * r)] = 1
        mask0 = mask0.reshape(-1)

        if mask is None:
            mask = mask0
        else:
            mask = mask * mask0
    
    if mask is not None:
        mask = (mask > 0.0).astype('float32')

    if (mask is None) or \
        (ratio <= 0.0) or \
        (mask.sum() == 0) or \
        ((1 - mask).sum() == 0):
        return np.random.choice(num_pixel, num_sample)

    if use_bbox:
        mask = mask.reshape(-1, width)
        x, y = np.where(mask == 1)
        mask = np.zeros_like(mask)
        mask[x.min(): x.max()+1, y.min(): y.max()+1] = 1.0
        mask = mask.reshape(-1)

    try:
        probs = mask * ratio / (mask.sum()) + (1 - mask) / (num_pixel - mask.sum()) * (1 - ratio)
        # x = np.random.choice(num_pixel, num_sample, True, p=probs)
        return np.random.choice(num_pixel, num_sample, True, p=probs)
    
    except Exception:
        return np.random.choice(num_pixel, num_sample)


def colormap(dz):
    return plt.cm.jet(dz)
    # return plt.cm.viridis(dz)
    # return plt.cm.gray(dz)


def recover_image(img, min_val=-1, max_val=1, width=512, bg=None, weight=None, raw=False):
    if raw: return img

    sizes = img.size()
    height = sizes[0] // width
    img = img.float().to('cpu')

    if len(sizes) == 1 and (bg is not None):
        bg_mask = img.eq(bg)[:, None].type_as(img)

    img = ((img - min_val) / (max_val - min_val)).clamp(min=0, max=1)
    if len(sizes) == 1:
        img = torch.from_numpy(colormap(img.numpy())[:, :3])
    if weight is not None:
        weight = weight.float().to('cpu')
        img = img * weight[:, None]

    if bg is not None:
        img = img * (1 - bg_mask) + bg_mask
    img = img.reshape(height, width, -1)
    return img

    
def write_images(writer, images, updates): 
    for tag in images:
        img = images[tag]
        tag, dataform = tag.split(':')
        writer.add_image(tag, img, updates, dataformats=dataform)


def compute_psnr(p, t):
    """Compute PSNR of model image predictions.
    :param prediction: Return value of forward pass.
    :param ground_truth: Ground truth.
    :return: (psnr, ssim): tuple of floats
    """
    ssim = skimage.metrics.structural_similarity(p, t, multichannel=True, data_range=1)
    psnr = skimage.metrics.peak_signal_noise_ratio(p, t, data_range=1)
    return ssim, psnr


def save_point_cloud(filename, xyz, rgb=None):
    if rgb is None:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    else:
        vertex = np.array([(xyz[k, 0], xyz[k, 1], xyz[k, 2], rgb[k, 0], rgb[k, 1], rgb[k, 2]) for k in range(xyz.shape[0])], 
            dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(filename)
    # from fairseq import pdb; pdb.set_trace()
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(open(filename, 'wb'))


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


class Timer(StopwatchMeter):
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start()
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.stop()


class GPUTimer(object):
    def __enter__(self):
        """Start a new timer as a context manager"""
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        self.sum = 0
        return self

    def __exit__(self, *exc_info):
        """Stop the context manager timer"""
        self.end.record()
        torch.cuda.synchronize()
        self.sum = self.start.elapsed_time(self.end) / 1000.
