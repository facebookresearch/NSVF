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
import sys, os
import argparse
import open3d as o3d

from fairseq import options
from fairseq import checkpoint_utils
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
    parser.add_argument('--user-dir', default='fairnr')
    args = options.parse_args_and_arch(parser)
    
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path], suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]
    voxel_idx, voxel_pts = model.encoder.extract_voxels()
    
    # write to ply file.
    points = [
        (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2], voxel_idx[k])
        for k in range(voxel_idx.size(0))
    ]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('quality', 'f4')])
    PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(os.path.join(args.output, args.name + '.ply'))
    
    # model = torch.load(args.path)
    # voxel_pts = model['model']['encoder.points'][model['model']['encoder.keep'].bool()]
    # from fairseq import pdb;pdb.set_trace()
    # # write to ply file.
    # points = [
    #     (voxel_pts[k, 0], voxel_pts[k, 1], voxel_pts[k, 2])
    #     for k in range(voxel_pts.size(0))
    # ]
    # vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    # PlyData([PlyElement.describe(vertex, 'vertex')], text=True).write(os.path.join(args.output, args.name + '.ply'))

if __name__ == '__main__':
    cli_main()
