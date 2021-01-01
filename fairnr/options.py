# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import torch


from fairseq import options


def parse_args_and_arch(*args, **kwargs):
    return options.parse_args_and_arch(*args, **kwargs)


def get_rendering_parser(default_task="single_object_rendering"):
    parser = options.get_parser("Rendering", default_task)
    options.add_dataset_args(parser, gen=True)
    add_rendering_args(parser)
    return parser


def add_rendering_args(parser):
    group = parser.add_argument_group("Rendering")
    options.add_common_eval_args(group)
    group.add_argument("--render-beam", default=5, type=int, metavar="N",
                       help="beam size for parallel rendering")
    group.add_argument("--render-resolution", default="512x512", type=str, metavar="N", help='if provide two numbers, means H x W')
    group.add_argument("--render-angular-speed", default=1, type=float, metavar="D",
                       help="angular speed when rendering around the object")
    group.add_argument("--render-num-frames", default=500, type=int, metavar="N")
    group.add_argument("--render-path-style", default="circle", choices=["circle", "zoomin_circle", "zoomin_line"], type=str)
    group.add_argument("--render-path-args", default="{'radius': 2.5, 'h': 0.0}",
                       help="specialized arguments for rendering paths")
    group.add_argument("--render-output", default=None, type=str)
    group.add_argument("--render-at-vector", default="(0,0,0)", type=str)
    group.add_argument("--render-up-vector", default="(0,0,-1)", type=str)
    group.add_argument("--render-output-types", nargs="+", type=str, default=["color"], 
                        choices=["target", "color", "depth", "normal", "voxel", "predn", "point", "featn2", "vcolors"])
    group.add_argument("--render-raymarching-steps", default=None, type=int)
    group.add_argument("--render-save-fps", default=24, type=int)
    group.add_argument("--render-combine-output", action='store_true', 
                       help="if set, concat the images into one file.")
    group.add_argument("--render-camera-poses", default=None, type=str,
                       help="text file saved for the testing trajectories")
    group.add_argument("--render-camera-intrinsics", default=None, type=str)
    group.add_argument("--render-views", type=str, default=None, 
                        help="views sampled for rendering, you can set specific view id, or a range")