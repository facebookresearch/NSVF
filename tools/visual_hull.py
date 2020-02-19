import os
import glob
import numpy as np
import torch
import cv2

from fairdr.data import RenderedImageDataset, WorldCoordDataset

# multi-view images
DIR = "/private/home/jgu/data/shapenet/maria"
dataset = RenderedImageDataset(DIR, 50, 50)

# visual-hull parameters
s = 128     # voxel resolution
extent = 5.
imgW, imgH = 1024, 1024
background = -0.498
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
points[:, 1] += 1
points = np.vstack((points.T, np.ones((1, nb_points_init))))

# space carving
print('build data ok.')
occupancy = np.zeros(points.shape[1]).astype(np.int)
packed_data, intrinsics = next(iter(dataset))

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

    image_mask = 1 - np.all(np.logical_and(
        data['rgb'] > (background - tau),
        data['rgb'] < (background + tau)), 0).reshape(imgW, imgH).astype(np.int)
    res = image_mask[sub_uvs[1, :], sub_uvs[0, :]]
    occupancy[indices] += res

def save_mdh(data,fname='out.mhd',ex=[1,1,1]):
    import vtk
    from vtk.util import numpy_support
    
    imdata = vtk.vtkImageData()
    imdata.SetDimensions(data.shape)
    imdata.SetSpacing([1,1,1])
    imdata.SetOrigin([0,0,0])
    depthArray = numpy_support.numpy_to_vtk(data.ravel(), deep=True, array_type=vtk.VTK_DOUBLE)
    imdata.GetPointData().SetScalars(depthArray)
    
    writer = vtk.vtkMetaImageWriter()
    writer.SetFileName(fname)
    writer.SetInputData(imdata)
    writer.Write()

occupancy = occupancy.reshape(s, s, s)
ox, oy, oz = (occupancy >= th).nonzero()          # sparse voxel indexs
ovoxels = points[:3, ox * s * s + oy * s + oz].T  # sparse voxel coords

# save data
fname = os.path.join(DIR, 'sparse_voxel.txt')
with open(fname, 'w') as f:
    for i in range(ox.shape[0]):
        print('{} {} {} {} {} {}'.format(
            ox[i], oy[i], oz[i], 
            ovoxels[i, 0], ovoxels[i, 1], ovoxels[i, 2]), 
        file=f)

# from sklearn.neighbors import KDTree
# tree = KDTree(ovoxels)
# ray-casting: find the intersection voxels
# coord_dataset = WorldCoordDataset(dataset)
# for data in next(iter(coord_dataset)):
#     not_done = np.ones_like(data['alpha'])
#     hits, deps = np.ones_like(not_done) * -1, np.zeros_like(not_done)
#     min_depth, max_depth = 2.2, 4.5
#     ray_start, ray_dir = data['ray_start'], data['ray_dir'].T
#     not_done = not_done.astype('bool')
#     deps = deps + min_depth
#     while not_done.any():
#         deps[not_done] += voxw
#         if deps.max() > max_depth:
#             break
#         print('raymarching depth: {}, not done: {}'.format(deps.max(), not_done.sum()))
#         ray = ray_start[None, :] + ray_dir[not_done] * deps[not_done][:, None]
#         idx, dis = tree.query_radius(ray, r=voxw, return_distance=True, sort_results=True)
#         done = np.array([len(i) > 0 for i in idx]).astype('bool')
#         hits[np.where(not_done)[0][done]] = np.array([i[0] for i in idx if len(i) > 0])
#         not_done[np.where(not_done)[0][done]] = False
#     cv2.imwrite("/private/home/jgu/work/Notebooks/test.png", 255 * (1 - deps.reshape(imgH, imgW) / max_depth))
#     from fairseq.pdb import set_trace; set_trace()
# # save(occupancy, fname='/checkpoint/jgu/space/out128.mhd')
# print('done')
