# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
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
    

class PosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, no_linear=False, scale=1024):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        half_dim = out_dim // 2 // in_dim
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        
        self.emb = nn.Parameter(emb, requires_grad=False)
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size()
        x = self.scale * x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            return self.linear(x)
        return x


class NeRFPosEmbLinear(nn.Module):

    def __init__(self, in_dim, out_dim, angular=False, no_linear=False, cat_input=False):
        super().__init__()
        assert out_dim % (2 * in_dim) == 0, "dimension must be dividable"
        L = out_dim // 2 // in_dim
        emb = torch.exp(torch.arange(L, dtype=torch.float) * math.log(2.))
        if not angular:
            emb = emb * math.pi

        self.emb = nn.Parameter(emb, requires_grad=False)
        self.angular = angular
        self.linear = Linear(out_dim, out_dim) if not no_linear else None
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cat_input = cat_input

    def forward(self, x):
        assert x.size(-1) == self.in_dim, "size must match"
        sizes = x.size() 
        inputs = x.clone()

        if self.angular:
            x = torch.acos(x.clamp(-1 + 1e-6, 1 - 1e-6))
        x = x.unsqueeze(-1) @ self.emb.unsqueeze(0)
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.view(*sizes[:-1], self.out_dim)
        if self.linear is not None:
            x = self.linear(x)
        if self.cat_input:
            x = torch.cat([x, inputs], -1)
        return x

    def extra_repr(self) -> str:
        outstr = 'Sinusoidal (in={}, out={}, angular={})'.format(
            self.in_dim, self.out_dim, self.angular)
        if self.cat_input:
            outstr = 'Cat({}, {})'.format(self.in_dim, outstr)
        return outstr


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


class FCBlock(nn.Module):
    def __init__(self,
                 hidden_ch,
                 num_hidden_layers,
                 in_features,
                 out_features,
                 outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features, hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(hidden_ch, hidden_ch))

        if outermost_linear:
            self.net.append(Linear(hidden_ch, out_features))
        else:
            self.net.append(FCLayer(hidden_ch, out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)

    def __getitem__(self,item):
        return self.net[item]

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)


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