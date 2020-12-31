# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn.functional as F

from fairnr.data import data_utils as D
try:
    from fairnr.clib._ext import build_octree
except ImportError:
    pass

INF = 1000.0


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
        return x / (l2 + 1e-8), l2

    else:
        l2 = np.linalg.norm(x, order, axis)
        l2 = np.expand_dims(l2, axis)
        l2[l2==0] = 1
        return x / l2, l2


def parse_extrinsics(extrinsics, world2camera=True):
    """ this function is only for numpy for now"""
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
        extrinsics = extrinsics.reshape(4, 4)
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


def r6d2mat(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalisation per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def get_ray_direction(ray_start, uv, intrinsics, inv_RT, depths=None):
    if depths is None:
        depths = 1
    rt_cam = uv2cam(uv, depths, intrinsics, True)       
    rt = cam2world(rt_cam, inv_RT)
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
    else:
        at = torch.tensor(at).type_as(camera_position)
    if up is None:
        up = torch.zeros_like(camera_position)
        up[2] = -1
    else:
        up = torch.tensor(up).type_as(camera_position)

    z_axis = normalize(at - camera_position)[0]
    x_axis = normalize(cross(up, z_axis))[0]
    y_axis = normalize(cross(z_axis, x_axis))[0]

    R = cat([x_axis[:, None], y_axis[:, None], z_axis[:, None]], axis=1)
    return R
    

def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def compute_normal_map(ray_start, ray_dir, depths, RT, width=512, proj=False):
    # TODO:
    # this function is pytorch-only (for not)
    wld_coords = ray(ray_start, ray_dir, depths.unsqueeze(-1)).transpose(0, 1)
    cam_coords = matmul(RT[:3, :3], wld_coords) + RT[:3, 3].unsqueeze(-1)
    cam_coords = D.unflatten_img(cam_coords, width)

    # estimate local normal
    shift_l = cam_coords[:, 2:,  :]
    shift_r = cam_coords[:, :-2, :]
    shift_u = cam_coords[:, :, 2: ]
    shift_d = cam_coords[:, :, :-2]
    diff_hor = normalize(shift_r - shift_l, axis=0)[0][:, :, 1:-1]
    diff_ver = normalize(shift_u - shift_d, axis=0)[0][:, 1:-1, :]
    normal = cross(diff_hor, diff_ver)
    _normal = normal.new_zeros(*cam_coords.size())
    _normal[:, 1:-1, 1:-1] = normal
    _normal = _normal.reshape(3, -1).transpose(0, 1)

    # compute the projected color
    if proj:
        _normal = normalize(_normal, axis=1)[0]
        wld_coords0 = ray(ray_start, ray_dir, 0).transpose(0, 1)
        cam_coords0 = matmul(RT[:3, :3], wld_coords0) + RT[:3, 3].unsqueeze(-1)
        cam_coords0 = D.unflatten_img(cam_coords0, width)
        cam_raydir = normalize(cam_coords - cam_coords0, 0)[0].reshape(3, -1).transpose(0, 1)
        proj_factor = (_normal * cam_raydir).sum(-1).abs() * 0.8 + 0.2
        return proj_factor
    return _normal


def trilinear_interp(p, q, point_feats):
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)
    point_feats = (weights * point_feats).sum(1)
    return point_feats


# helper functions for encoder

def padding_points(xs, pad):
    if len(xs) == 1:
        return xs[0].unsqueeze(0)
    
    maxlen = max([x.size(0) for x in xs])
    xt = xs[0].new_ones(len(xs), maxlen, xs[0].size(1)).fill_(pad)
    for i in range(len(xs)):
        xt[i, :xs[i].size(0)] = xs[i]
    return xt


def pruning_points(feats, points, scores, depth=0, th=0.5):
    if depth > 0:
        g = int(8 ** depth)
        scores = scores.reshape(scores.size(0), -1, g).sum(-1, keepdim=True)
        scores = scores.expand(*scores.size()[:2], g).reshape(scores.size(0), -1)
    alpha = (1 - torch.exp(-scores)) > th
    feats = [feats[i][alpha[i]] for i in range(alpha.size(0))]
    points = [points[i][alpha[i]] for i in range(alpha.size(0))]
    points = padding_points(points, INF)
    feats = padding_points(feats, 0)
    return feats, points


def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    ox, oy, oz = torch.meshgrid([c, c, c])
    offset = (torch.cat([
                    ox.reshape(-1, 1), 
                    oy.reshape(-1, 1), 
                    oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    if not offset_only:
        return point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel
    return offset.type_as(point_xyz) * quarter_voxel


def discretize_points(voxel_points, voxel_size):
    # this function turns voxel centers/corners into integer indeices
    # we assume all points are alreay put as voxels (real numbers)
    minimal_voxel_point = voxel_points.min(dim=0, keepdim=True)[0]
    voxel_indices = ((voxel_points - minimal_voxel_point) / voxel_size).round_().long()  # float
    residual = (voxel_points - voxel_indices.type_as(voxel_points) * voxel_size).mean(0, keepdim=True)
    return voxel_indices, residual


def splitting_points(point_xyz, point_feats, values, half_voxel):        
    # generate new centers
    quarter_voxel = half_voxel * .5
    new_points = offset_points(point_xyz, quarter_voxel).reshape(-1, 3)
    old_coords = discretize_points(point_xyz, quarter_voxel)[0]
    new_coords = offset_points(old_coords).reshape(-1, 3)
    new_keys0  = offset_points(new_coords).reshape(-1, 3)
    
    # get unique keys and inverse indices (for original key0, where it maps to in keys)
    new_keys, new_feats = torch.unique(new_keys0, dim=0, sorted=True, return_inverse=True)
    new_keys_idx = new_feats.new_zeros(new_keys.size(0)).scatter_(
        0, new_feats, torch.arange(new_keys0.size(0), device=new_feats.device) // 64)
    
    # recompute key vectors using trilinear interpolation 
    new_feats = new_feats.reshape(-1, 8)
    
    if values is not None:
        p = (new_keys - old_coords[new_keys_idx]).type_as(point_xyz).unsqueeze(1) * .25 + 0.5 # (1/4 voxel size)
        q = offset_points(p, .5, offset_only=True).unsqueeze(0) + 0.5   # BUG?
        point_feats = point_feats[new_keys_idx]
        point_feats = F.embedding(point_feats, values).view(point_feats.size(0), -1)
        new_values = trilinear_interp(p, q, point_feats)
    else:
        new_values = None
    return new_points, new_feats, new_values, new_keys


def expand_points(voxel_points, voxel_size):
    _voxel_size = min([
        torch.sqrt(((voxel_points[j:j+1] - voxel_points[j+1:]) ** 2).sum(-1).min())
        for j in range(100)])
    depth = int(np.round(torch.log2(_voxel_size / voxel_size)))
    if depth > 0:
        half_voxel = _voxel_size / 2.0
        for _ in range(depth):
            voxel_points = offset_points(voxel_points, half_voxel / 2.0).reshape(-1, 3)
            half_voxel = half_voxel / 2.0
    
    return voxel_points, depth


def get_edge(depth_pts, voxel_pts, voxel_size, th=0.05):
    voxel_pts = offset_points(voxel_pts, voxel_size / 2.0)
    diff_pts = (voxel_pts - depth_pts[:, None, :]).norm(dim=2)
    ab = diff_pts.sort(dim=1)[0][:, :2]
    a, b = ab[:, 0], ab[:, 1]
    c = voxel_size
    p = (ab.sum(-1) + c) / 2.0
    h = (p * (p - a) * (p - b) * (p - c)) ** 0.5 / c
    return h < (th * voxel_size)


# fill-in image
def fill_in(shape, hits, input, initial=1.0):
    input_sizes = [k for k in input.size()]
    if (len(input_sizes) == len(shape)) and \
        all([shape[i] == input_sizes[i] for i in range(len(shape))]):
        return input   # shape is the same no need to fill
        
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    if input is not None:
        if len(shape) == 1:
            return output.masked_scatter(hits, input)
        return output.masked_scatter(hits.unsqueeze(-1).expand(*shape), input)
    return output


def build_easy_octree(points, half_voxel):
    coords, residual = discretize_points(points, half_voxel)
    ranges = coords.max(0)[0] - coords.min(0)[0]
    depths = torch.log2(ranges.max().float()).ceil_().long() - 1
    center = (coords.max(0)[0] + coords.min(0)[0]) / 2
    centers, children = build_octree(center, coords, int(depths))
    centers = centers.float() * half_voxel + residual   # transform back to float
    return centers, children


def cartesian_to_spherical(xyz):
    """ xyz: batch x 3
    """
    r = xyz.norm(p=2, dim=-1)
    theta = torch.atan2(xyz[:, :2].norm(p=2, dim=-1), xyz[:, 2])
    phi = torch.atan2(xyz[:, 1], xyz[:, 0])
    return torch.stack((r, theta, phi), 1)


def spherical_to_cartesian(rtp):
    x = rtp[:, 0] * torch.sin(rtp[:, 1]) * torch.cos(rtp[:, 2])
    y = rtp[:, 0] * torch.sin(rtp[:, 1]) * torch.sin(rtp[:, 2])
    z = rtp[:, 0] * torch.cos(rtp[:, 1])
    return torch.stack((x, y, z), 1)