#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import scipy.io
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

# Load camera matrices
data = scipy.io.loadmat("data/dino_Ps.mat")
data = data["P"]
projections = [data[0, i] for i in range(data.shape[1])]

# load images
files = sorted(glob.glob("data/*.ppm"))
images = []
for f in files:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(float)
    im /= 255
    images.append(im[:, :, ::-1])
    

# get silouhette from images
imgH, imgW, __ = images[0].shape
silhouette = []
for im in images:
    temp = np.abs(im - [0.0, 0.0, 0.75])
    temp = np.sum(temp, axis=2)
    y, x = np.where(temp <= 1.1)
    im[y, x, :] = [0.0, 0.0, 0.0]
    im = im[:, :, 0]
    im[im > 0] = 1.0
    im = im.astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    silhouette.append(im)

#    plt.figure()
#    plt.imshow(im)
    
#%%
# create voxel grid
s = 120
x, y, z = np.mgrid[:s, :s, :s]
pts = np.vstack((x.flatten(), y.flatten(), z.flatten())).astype(float)
pts = pts.T
nb_points_init = pts.shape[0]
xmax, ymax, zmax = np.max(pts, axis=0)
pts[:, 0] /= xmax
pts[:, 1] /= ymax
pts[:, 2] /= zmax
center = pts.mean(axis=0)
pts -= center
pts /= 5
pts[:, 2] -= 0.62

pts = np.vstack((pts.T, np.ones((1, nb_points_init))))

filled = []
for P, im in zip(projections, silhouette):
    uvs = P @ pts
    uvs /= uvs[2, :]
    uvs = np.round(uvs).astype(int)
    x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < imgW)
    y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < imgH)
    good = np.logical_and(x_good, y_good)
    indices = np.where(good)[0]
    fill = np.zeros(uvs.shape[1])
    sub_uvs = uvs[:2, indices]
    res = im[sub_uvs[1, :], sub_uvs[0, :]]
    fill[indices] = res 
    
    filled.append(fill)
filled = np.vstack(filled)

# the occupancy is computed as the number of camera in which the point "seems" not empty
occupancy = np.sum(filled, axis=0)

# Select occupied voxels
pts = pts.T
good_points = pts[occupancy > 4, :]
    

        
#%% save point cloud with occupancy scalar 
filename = "shape.txt"
with open(filename, "w") as fout:
    fout.write("x,y,z,occ\n")
    for occ, p in zip(occupancy, pts[:, :3]):
        fout.write(",".join(p.astype(str)) + "," + str(occ) + "\n")
        




    
#%% save as rectilinear grid (this enables paraview to display its iso-volume as a mesh)
import vtk

xCoords = vtk.vtkFloatArray()
x = pts[::s*s, 0]
y = pts[:s*s:s, 1]
z = pts[:s, 2]
for i in x:
    xCoords.InsertNextValue(i)
yCoords = vtk.vtkFloatArray()
for i in y:
    yCoords.InsertNextValue(i)
zCoords = vtk.vtkFloatArray()
for i in z:
    zCoords.InsertNextValue(i)
values = vtk.vtkFloatArray()
for i in occupancy:
    values.InsertNextValue(i)
rgrid = vtk.vtkRectilinearGrid()
rgrid.SetDimensions(len(x), len(y), len(z))
rgrid.SetXCoordinates(xCoords)
rgrid.SetYCoordinates(yCoords)
rgrid.SetZCoordinates(zCoords)
rgrid.GetPointData().SetScalars(values)

writer = vtk.vtkXMLRectilinearGridWriter()
writer.SetFileName("shape.vtr")
writer.SetInputData(rgrid)
writer.Write()