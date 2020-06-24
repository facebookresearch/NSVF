#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This code is used for extact voxels/meshes from the learne model
"""
import logging
import numpy as np
import torch
import sys
import argparse
import open3d as o3d
from plyfile import PlyData, PlyElement

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairnr_cli.extract')


def cli_main():
    parser = argparse.ArgumentParser(description='Extract geometry from a trained model (only for learnable embeddings).')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--name', type=str, default='sparsevoxel')
    args = parser.parse_args()

    state = torch.load(args.path)['model']
    keep = state['field.backbone.keep'].bool()
    points = state['field.backbone.points'][keep].tolist()
    voxidx = torch.arange(keep.size(0))[keep].tolist()

    # write to ply file.
    points = [tuple(points[k] + [voxidx[k]]) for k in range(len(voxidx))]    
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(args.output + '/some_binary.ply')
    # from fairseq import pdb; pdb.set_trace()
    # voxel_size = state['field.VOXEL_SIZE']
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points.numpy())
    # o3d.io.write_point_cloud("{}/{}_{:.4f}.ply".format(args.output, args.name, voxel_size), pcd)
    

if __name__ == '__main__':
    cli_main()
