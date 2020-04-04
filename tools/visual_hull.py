import os, sys
import glob
import numpy as np
import torch
import cv2
import argparse
import open3d as o3d
from fairdr.data import ShapeViewDataset, WorldCoordDataset

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--dir', type=str, required=True)
parser.add_argument('--frames', type=int, default=50)
parser.add_argument('--voxel_res', type=int, default=128)
parser.add_argument('--image_res', type=int, default=1024)
parser.add_argument('--image_res_H', type=int, default=None)
parser.add_argument('--extent', type=float, default=5.0)
parser.add_argument('--marching_cube', action='store_true')
parser.add_argument('--downsample', type=float, default=4.0)
parser.add_argument('--load_mask', action='store_true')
parser.add_argument('--boundingbox', action='store_true')
args = parser.parse_args()


# multi-view images
# DIR = "/private/home/jgu/data/shapenet/bunny/bunny/{}".format(sys.argv[1])
DIR = args.dir

print(DIR)

# DIR = "/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug"
dataset = ShapeViewDataset(
    DIR, args.frames, args.frames, 
    load_mask=args.load_mask, binarize=False,
    resolution=None)

# visual-hull parameters
s = args.voxel_res     # voxel resolution
extent = args.extent
imgW = args.image_res
if args.image_res_H is None:
    imgH = args.image_res
else:
    imgH = args.image_res_H

background = -1
tau = 0.005
th = args.frames - 1
voxw = extent / s
voxd = args.downsample

# prepare voxel grids
x, y, z = np.mgrid[:s, :s, :s]
points = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(np.float32)
points = points.T
nb_points_init = points.shape[0]
xmax, ymax, zmax = np.max(points, axis=0)
points[:, 0] /= xmax
points[:, 1] /= ymax
points[:, 2] /= zmax
center = points.mean(axis=0)
points -= center
points *= extent

# # HACK: do we need this ?
# points[:, 1] += 1
points = np.vstack((points.T, np.ones((1, nb_points_init))))

# space carving
print('build data ok.')
occupancy = np.zeros(points.shape[1]).astype(np.int)
index, u_data, packed_data = next(iter(dataset))
intrinsics = u_data['intrinsics']

for i, data in enumerate(packed_data):
    print('process view: {}'.format(i))
    P = intrinsics @ np.linalg.inv(data['extrinsics'])
    P = P[:3, :]

    uvs = P @ points
    uvs = uvs / uvs[2, :]
    uvs = np.round(uvs).astype(int)
    x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
    y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
    good = np.logical_and(x_good, y_good)
    indices = np.where(good)[0]
    sub_uvs = uvs[:2, indices]

    # image_mask = 1 - np.all(np.logical_and(
    #     data['rgb'] > (background - tau),
    #     data['rgb'] < (background + tau)), 0).reshape(imgW, imgH).astype(np.int)
    if not args.load_mask:
        image_mask = data['alpha'].reshape(imgH, imgW).astype(np.int)
    else:
        image_mask = data['mask'].reshape(imgH, imgW).astype(np.int)

    res = image_mask[sub_uvs[1, :], sub_uvs[0, :]]
    occupancy[indices] += res

occupancy = occupancy.reshape(s, s, s)

def save_mesh(verts, faces, normals, fname):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_triangle_mesh(fname, mesh)

def downsample_pcd(verts, voxel_size, fname):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    downpcd = pcd.voxel_down_sample(voxel_size)
    o3d.io.write_point_cloud(fname, downpcd)
    return np.asarray(downpcd.points)

def voxelize_pcd(verts, voxel_size, fname):
    
    pcd = o3d.geometry.PointCloud()
    ovoxels = np.floor(verts / voxel_size) * voxel_size + (voxel_size / 2.)
    
    # remove duplicates
    x = np.random.rand(ovoxels.shape[1])
    y = ovoxels.dot(x)
    unique, index = np.unique(y, return_index=True)
    ovoxels = ovoxels[index]

    pcd.points = o3d.utility.Vector3dVector(ovoxels)
    o3d.io.write_point_cloud(fname, pcd)
    return ovoxels

def voxelize_bbx(verts, voxel_size, fname):
    xyz_min, xyz_max = verts.min(0) - voxd * .5, verts.max(0) + voxel_size * .5
    x, y, z = np.mgrid[
        range(int((xyz_max[0] - xyz_min[0]) / voxel_size) + 1),
        range(int((xyz_max[1] - xyz_min[1]) / voxel_size) + 1),
        range(int((xyz_max[2] - xyz_min[2]) / voxel_size) + 1)
    ]
    ovoxels = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(np.float32)
    ovoxels = ovoxels.T
    ovoxels = ovoxels * voxel_size + xyz_min[None, :]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ovoxels)
    o3d.io.write_point_cloud(fname, pcd)
    return ovoxels

if not args.marching_cube:
    from fairseq import pdb; pdb.set_trace()
    ox, oy, oz = (occupancy >= th).nonzero()        # sparse voxel indexs
    verts = points[:3, ox * s * s + oy * s + oz].T  # sparse voxel coords
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    o3d.io.write_point_cloud(os.path.join(DIR, 'visualhull.ply'), pcd)
else:
    from skimage import measure
    verts, faces, normals, values = measure.marching_cubes_lewiner(occupancy, th)
    verts = (verts - args.voxel_res / 2) / args.voxel_res * extent
    save_mesh(verts, faces, normals, os.path.join(DIR, 'visualhull_mc.ply'))
    # verts = downsample_pcd(verts, voxd, os.path.join(DIR, 'visualhull_down.ply'))

if args.boundingbox:
    ovoxels = voxelize_bbx(verts, voxd, os.path.join(DIR, 'visualhull_bbox{}.ply'.format(voxd)))
else:
    ovoxels = voxelize_pcd(verts, voxd, os.path.join(DIR, 'visualhull_voxel{}.ply'.format(voxd)))

o = np.floor(ovoxels / voxd).astype(int)
ox, oy, oz = o[:, 0], o[:, 1], o[:, 2]

# save data
fname = os.path.join(DIR, 'voxel.txt')
with open(fname, 'w') as f:
    for i in range(ox.shape[0]):
        print('{} {} {} {} {} {}'.format(
            ox[i], oy[i], oz[i], 
            ovoxels[i, 0], ovoxels[i, 1], ovoxels[i, 2]), 
        file=f)

