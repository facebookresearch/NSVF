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

from fairseq import options
from fairseq import checkpoint_utils
from plyfile import PlyData, PlyElement


def main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairnr_cli.extract')
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path], suffix=getattr(args, "checkpoint_suffix", ""))
    model = models[0]
    if use_cuda:
        model.cuda()

    if args.format == 'mc_mesh':
        plydata = model.encoder.export_surfaces(
            model.field, th=args.mc_threshold, 
            bits=2 * args.mc_num_samples_per_halfvoxel)
    elif args.format == 'voxel_center':
        plydata = model.encoder.export_voxels(False)
    elif args.format == 'voxel_mesh':
        plydata = model.encoder.export_voxels(True)
    else:
        raise NotImplementedError

    # write to ply file.
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    plydata.text = args.savetext
    plydata.write(open(os.path.join(args.output, args.name + '.ply'), 'wb'))


def cli_main():
    parser = argparse.ArgumentParser(description='Extract geometry from a trained model (only for learnable embeddings).')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--name', type=str, default='sparsevoxel')
    parser.add_argument('--format', type=str, choices=['voxel_center', 'voxel_mesh', 'mc_mesh'])
    parser.add_argument('--savetext', action='store_true', help='save .ply in plain text')
    parser.add_argument('--mc-num-samples-per-halfvoxel', type=int, default=8,
                        help="""the number of point samples every half voxel-size for marching cube. 
                        For instance, by setting to 8, it will use (8 x 2) ^ 3 = 4096 points to compute density for each voxel.
                        In practise, the larger this number is, the more accurate surface you get.
                        """)
    parser.add_argument('--mc-threshold', type=float, default=0.5, 
                        help="""the threshold used to find the isosurface from the learned implicit field.
                        In our implementation, we define our values as ``1 - exp(-max(0, density))`` 
                        where "0" is empty and "1" is fully occupied.
                        """)
    parser.add_argument('--user-dir', default='fairnr')
    parser.add_argument('--cpu', action='store_true')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
