# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F

TINY = 1e-7


def rgb_loss(predicts, rgbs, masks, L1=False, sum=False):
    if masks.sum() == 0:
        loss = predicts.new_zeros(1)

    elif L1:
        loss = torch.abs(predicts[masks] - rgbs[masks]).sum(-1)
    
    else:
        loss = ((predicts[masks] - rgbs[masks]) ** 2).sum(-1)

    return loss.mean() if not sum else loss.sum()


def space_loss(occupancy, masks, sum=False):
    if masks.sum() == 0:
        loss = occupancy.new_zeros(1)
    else:
        loss = F.binary_cross_entropy(occupancy[masks], torch.zeros_like(occupancy[masks]), reduction='none')

    return loss.mean() if not sum else loss.sum()


def occupancy_loss(occupancy, masks, sum=False):
    if masks.sum() == 0:
        loss = occupancy.new_zeros(1)
    else:
        loss = F.binary_cross_entropy(occupancy[masks], torch.ones_like(occupancy[masks]), reduction='none')

    return loss.mean() if not sum else loss.sum()


def depth_regularization_loss(depths, sum=False):
    loss = (torch.min(depths, torch.zeros_like(depths)) ** 2) * 10000.0
    return loss.mean() if not sum else loss.sum()


def depth_loss(depths, depth_gt, masks, sum=False):
    if masks.sum() == 0:
        loss = depths.new_zeros(1)
    else:
        loss = ((depths[masks] - depth_gt[masks]) ** 2)
    
    return loss.mean() if not sum else loss.sum()