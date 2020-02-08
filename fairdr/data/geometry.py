# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


def parse_intrinsics(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return fx, fy, cx, cy


def uv2cam(uv, z, intrinsics, homogeneous=False):
    fx, fy, cx, cy = parse_intrinsics(intrinsics)
    x_lift = (uv[0] - cx) / fx * z
    y_lift = (uv[1] - cy) / fy * z
    z_lift = np.ones_like(x_lift) * z

    if homogeneous:
        return np.stack([x_lift, y_lift, z_lift, np.ones_like(z_lift)])
    else:
        return np.stack([x_lift, y_lift, z_lift])


def cam2world(xyz_cam, inv_RT):
    return np.matmul(inv_RT, xyz_cam)[:3]


def normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)
