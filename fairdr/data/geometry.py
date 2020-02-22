# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from fairdr.data import data_utils as D


def ones_like(x):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.ones_like(x)


def stack(x):
    T = torch if isinstance(x[0], torch.Tensor) else np
    return T.stack(x)


def matmul(x, y):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.matmul(x, y)


def cross(x, y, axis=0):
    T = torch if isinstance(x, torch.Tensor) else np
    return T.cross(x, y, axis)


def cat(x, axis=1):
    if isinstance(x[0], torch.Tensor):
        return torch.cat(x, dim=axis)
    return np.concatenate(x, axis=axis)


def normalize(x, axis=-1, order=2):
    if isinstance(x, torch.Tensor):
        l2 = x.norm(p=order, dim=axis, keepdim=True)        
    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)

    l2[l2==0] = 1
    return x / l2, l2


def parse_extrinsics(extrinsics, world2camera=True):
    """ this function is only for numpy for now"""
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
    z_lift = ones_like(x_lift) * z

    if homogeneous:
        return stack([x_lift, y_lift, z_lift, ones_like(z_lift)])
    else:
        return stack([x_lift, y_lift, z_lift])


def cam2world(xyz_cam, inv_RT):
    return matmul(inv_RT, xyz_cam)[:3]


def get_ray_direction(ray_start, uv, intrinsics, inv_RT, depths=None):
    if depths is None:
        depths = 1
    rt_cam = uv2cam(uv, depths, intrinsics, True)       
    rt = cam2world(rt_cam, inv_RT)
    # from fairseq.pdb import set_trace; set_trace()
    ray_dir, _ = normalize(rt - ray_start[:, None], axis=0)
    return ray_dir


def look_at_rotation(camera_position, at=None, up=None, inverse=False, cv=False):
    """
    This function takes a vector 'camera_position' which specifies the location
    of the camera in world coordinates and two vectors `at` and `up` which
    indicate the position of the object and the up directions of the world
    coordinate system respectively. The object is assumed to be centered at
    the origin.

    The output is a rotation matrix representing the transformation
    from world coordinates -> view coordinates.

    Input:
        camera_position: 3
        at: 1 x 3 or N x 3  (0, 0, 0) in default
        up: 1 x 3 or N x 3  (0, 1, 0) in default
    """
    if at is None:
        at = torch.zeros_like(camera_position)

    if up is None:
        up = torch.zeros_like(camera_position)
        up[1] = 1

    z_axis = normalize(camera_position - at)[0]
    x_axis = normalize(cross(-z_axis, up))[0]
    y_axis = normalize(cross(x_axis, -z_axis))[0]
    R = cat([x_axis[None, :], -y_axis[None, :], -z_axis[None, :]], axis=0)
    R = R.transpose(0, 1)      # world --> camera

    # if cv:
    #     R_cam2cv = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]], 
    #                     device=R.device, dtype=R.dtype)
    #     R = R_cam2cv @ R    # world --> view
    return R.inverse()


def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def compute_normal_map(ray_start, ray_dir, depths, RT):
    # TODO:
    # this function is pytorch-only (for not)
    wld_coords = ray(ray_start, ray_dir, depths.unsqueeze(-1)).transpose(0, 1)
    cam_coords = matmul(RT[:3, :3], wld_coords) + RT[:3, 3].unsqueeze(-1)
    cam_coords = D.unflatten_img(cam_coords)

    # estimate local normal
    shift_l = cam_coords[:, 2:,  :]
    shift_r = cam_coords[:, :-2, :]
    shift_u = cam_coords[:, :, 2: ]
    shift_d = cam_coords[:, :, :-2]
    diff_hor = normalize(shift_r - shift_l, axis=0)[0][:, :, 1:-1]
    diff_ver = normalize(shift_u - shift_d, axis=0)[0][:, 1:-1, :]
    normal = cross(diff_hor, diff_ver).reshape(3, -1).transpose(0, 1)
    return normal