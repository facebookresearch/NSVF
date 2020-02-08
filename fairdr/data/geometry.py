# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Useful geometry operations (experimental): 
Main reference:
https://github.com/vsitzmann/scene-representation-networks/blob/master/geometry.py

In the future, I may replace it with https://pytorch3d.org/
"""

import numpy as np
import torch

from torch.nn import functional as F
import util


def parse_intrinsics(intrinsics):
    intrinsics = intrinsics.cuda()

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    return fx, fy, cx, cy