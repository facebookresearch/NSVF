# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

TINY = 1e-7


def rgb_loss(predicts, rgbs, masks=None, L1=False, sum=False):
    if masks is not None:
        if masks.sum() == 0:
            return predicts.new_zeros(1).mean()
        predicts = predicts[masks]
        rgbs = rgbs[masks]

    if L1:
        loss = torch.abs(predicts - rgbs).sum(-1)
    else:
        loss = ((predicts - rgbs) ** 2).sum(-1)
       
    return loss.mean() if not sum else loss.sum()


def depth_loss(depths, depth_gt, masks=None, sum=False):
    if masks is not None:
        if masks.sum() == 0:
            return depths.new_zeros(1).mean()
        depth_gt = depth_gt[masks]
        depths = depths[masks]

    loss = (depths[masks] - depth_gt[masks]) ** 2
    return loss.mean() if not sum else loss.sum()