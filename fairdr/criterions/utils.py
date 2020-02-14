# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

TINY = 1e-7


def rgb_loss(predicts, rgbs, masks, L1=False):
    if masks.sum() == 0:
        return predicts.new_zeros(1).sum()
    elif L1:
        return torch.abs(predicts[masks] - rgbs[masks]).sum(-1).mean()
    else:
        return ((predicts[masks] - rgbs[masks]) ** 2).sum(-1).mean()


def space_loss(occupancy, masks):
    if masks.sum() == 0:
        return occupancy.new_zeros(1).sum()
    return F.binary_cross_entropy(occupancy[masks], torch.zeros_like(occupancy[masks]))


def occupancy_loss(occupancy, masks):
    if masks.sum() == 0:
        return occupancy.new_zeros(1).sum()
    return F.binary_cross_entropy(occupancy[masks], torch.ones_like(occupancy[masks]))


def depth_regularization_loss(depths):
    neg_penalty = (torch.min(depths, torch.zeros_like(depths)) ** 2)
    return torch.mean(neg_penalty) * 10000.0


def depth_loss(depths, depth_gt, masks):
    return ((depths[masks] - depth_gt[masks]) ** 2).mean()