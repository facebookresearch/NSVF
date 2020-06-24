import os, sys
import glob
import numpy as np
import torch
import cv2

from fairnr.data import ShapeViewDataset, WorldCoordDataset


# multi-view images
# DIR = "/private/home/jgu/data/shapenet/bunny/bunny/{}".format(sys.argv[1])
DIR = "/private/home/jgu/data/shapenet/lucy2/lucy/{}".format(sys.argv[1])
print(DIR)

# DIR = "/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug"
dataset = ShapeViewDataset(DIR, 50, 50, load_mask=False, binarize=False)

# visual-hull parameters
s = 128     # voxel resolution
extent = 5.
imgW, imgH = 1024, 1024
background = -1
tau = 0.005
th = 49
voxw = extent / s

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
    image_mask = data['alpha'].reshape(imgW, imgH).astype(np.int)
    res = image_mask[sub_uvs[1, :], sub_uvs[0, :]]
    occupancy[indices] += res

occupancy = occupancy.reshape(s, s, s)
ox, oy, oz = (occupancy >= th).nonzero()          # sparse voxel indexs
ovoxels = points[:3, ox * s * s + oy * s + oz].T  # sparse voxel coords

# save data
fname = os.path.join(DIR, 'voxel.txt')
with open(fname, 'w') as f:
    for i in range(ox.shape[0]):
        print('{} {} {} {} {} {}'.format(
            ox[i], oy[i], oz[i], 
            ovoxels[i, 0], ovoxels[i, 1], ovoxels[i, 2]), 
        file=f)
