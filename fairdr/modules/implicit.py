# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F

from .linear import FCLayer, ResFCLayer, Linear


class ImplicitField(nn.Module):
    
    """
    An implicit field is a neural network that outputs a vector given any query point.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.fc_in  = Linear(args.input_features, args.ffn_embed_dim)
        self.fc_out = Linear(args.ffn_embed_dim, args.output_features)
        self.layers = nn.ModuleList([])
        self.layers.extend([
                ResFCLayer(args.ffn_embed_dim, args.ffn_embed_dim, args.ffn_hidden_dim, 'relu')
                if args.use_residual else
                FCLayer(args.ffn_embed_layer, args.ffn_embed_layer, 'relu')
            for _ in range(args.ffn_num_layers)
        ])
        self.num_layers = args.ffn_num_layers
    
    def forward(self, x):
        y = self.fc_in(x)
        for layer in self.layers:
            y = layer(y)

        # return final representation and prediction
        return y, F.sigmoid(self.fc_out(y))  