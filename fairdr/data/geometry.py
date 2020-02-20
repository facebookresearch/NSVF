# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch


# ----- numpy functions ------ #
def parse_extrinsics(extrinsics, world2camera=True):
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if world2camera:
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
    return extrinsics
    

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
    return a / np.expand_dims(l2, axis), l2


def look_at_rotation(camera_position, at=((0, 0, 0),), up=((0, 1, 0),)):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.
    """

    


# ----- pytorch functions ------ #

def ray(ray_start, ray_dir, depths):
    if ray_start.dim() + 1 == ray_dir.dim():
        ray_start = ray_start.unsqueeze(-2)
    if depths.dim() + 1 == ray_dir.dim():
        depths = depths.unsqueeze(-1)
    return ray_start + ray_dir * depths