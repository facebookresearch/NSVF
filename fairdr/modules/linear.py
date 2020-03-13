# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    return m
    

class FCLayer(nn.Module):
    """
    Reference:
        https://github.com/vsitzmann/pytorch_prototyping/blob/10f49b1e7df38a58fd78451eac91d7ac1a21df64/pytorch_prototyping.py
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm([out_dim]),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x) 


class ResFCLayer(nn.Module):
    """
    Reference:
        https://github.com/autonomousvision/occupancy_networks/blob/master/im2mesh/layers.py
    """
    def __init__(self, in_dim, out_dim, hidden_dim, act='relu', dropout=0.0):
        super().__init__()

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        # self.layernorm = LayerNorm(out_dim)
        self.nonlinear = get_activation_fn(activation=act)
        self.dropout = dropout

        # Initialization (?)
        nn.init.zeros_(self.fc2.weight)

    def forward(self, x):
        residual = x
        x = self.fc1(self.nonlinear(x))
        x = self.fc2(self.nonlinear(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        # return self.layernorm(x)
        return x