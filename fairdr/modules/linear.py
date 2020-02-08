# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from fairseq.modules import LayerNorm
from fairseq.utils import get_activation_fn


def Linear(in_features, out_features, bias=True, kaiming=True):
    m = nn.Linear(in_features, out_features, bias)
    if kaiming:
        nn.init.kaiming_normal_(m.weight, a=0, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight)
    
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class FCLayer(nn.Module):

    def __init__(self, in_dim, out_dim, act='relu', dropout=0.0):
        super().__init__()

        self.fc = Linear(in_dim, out_dim)
        self.layernorm = LayerNorm(out_dim)
        self.nonlinear = get_activation_fn(activation=act)
        self.dropout = dropout

    def forward(self, x):
        x = self.nonlinear(self.fc(x))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layernorm(x)


class ResFCLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, hidden_dim, act='relu', dropout=0.0):
        super().__init__()

        self.fc1 = Linear(in_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, out_dim)
        self.layernorm = LayerNorm(out_dim)
        self.nonlinear = get_activation_fn(activation=act)
        self.dropout = dropout

    def forward(self, x):
        residual = x
        x = self.fc2(self.nonlinear(self.fc1(x)))
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        return self.layernorm(x)