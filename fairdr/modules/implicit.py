# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import get_activation_fn
from .linear import FCLayer, ResFCLayer


class ImplicitField(nn.Module):
    
    """
    An implicit field is a neural network that outputs a vector given any query point.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.net = []
        self.net.append(FCLayer(args.input_features, args.ffn_embed_dim))
        for _ in range(args.ffn_num_layers):
            self.net.append(FCLayer(args.ffn_embed_dim, args.ffn_embed_dim))
        self.net.append(FCLayer(args.ffn_embed_dim, args.output_features))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights) 

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, x):
        return self.net(x)

class PixelRenderer(nn.Module):
    """
    Pixel generator based on 1x1 conv networks
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.net = []
        self.net.append(FCLayer(
            in_dim=args.renderer_in_features, 
            out_dim=args.renderer_hidden_dim))

        for _ in range(args.renderer_num_layers):
            self.net.append(FCLayer(
                in_dim=args.renderer_hidden_dim, 
                out_dim=args.renderer_hidden_dim))

        self.net.append(nn.Linear(
            in_features=args.renderer_hidden_dim, 
            out_features=3))
        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights) 

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, input):
        return self.net(input)