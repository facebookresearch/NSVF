# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import get_activation_fn
from fairdr.modules.utils import FCLayer, ResFCLayer


class ImplicitField(nn.Module):
    
    """
    An implicit field is a neural network that outputs a vector given any query point.
    """
    def __init__(self, args, in_dim, out_dim, hidden_dim, num_layers, outmost_linear=False):
        super().__init__()
        self.args = args
        
        self.net = []
        self.net.append(FCLayer(in_dim, hidden_dim))
        for _ in range(num_layers):
            self.net.append(FCLayer(hidden_dim, hidden_dim))

        if not outmost_linear:
            self.net.append(FCLayer(hidden_dim, out_dim))
        else:
            self.net.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*self.net)
        self.net.apply(self.init_weights)         

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')

    def forward(self, x):
        return self.net(x)


class SignedDistanceField(nn.Module):

    def __init__(self, args, in_dim, hidden_dim, recurrent=False):
        super().__init__()
        self.args = args
        self.recurrent = recurrent

        if recurrent:
            self.hidden_layer = nn.LSTMCell(input_size=in_dim, hidden_size=hidden_dim)
            self.hidden_layer.apply(init_recurrent_weights)
            lstm_forget_gate_init(self.hidden_layer)
        else:
            self.hidden_layer = FCLayer(in_dim, hidden_dim)

        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x, state=None):
        if self.recurrent:
            shape = x.size()
            state = self.hidden_layer(x.view(-1, shape[-1]), state)
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-5, max=5))
            
            return self.output_layer(state[0].view(*shape[:-1], -1)).squeeze(-1), state

        else:
            
            return self.output_layer(self.hidden_layer(x)).squeeze(-1), None


class TextureField(ImplicitField):
    """
    Pixel generator based on 1x1 conv networks
    """
    def __init__(self, args, in_dim, hidden_dim, num_layers):
        super().__init__(args, in_dim, 3, hidden_dim, num_layers, outmost_linear=True)


class OccupancyField(ImplicitField):
    """
    Occupancy Network which predicts 0~1 at every space
    """
    def __init__(self, args, in_dim, hidden_dim, num_layers):
        super().__init__(args, in_dim, 1, hidden_dim, num_layers, outmost_linear=True)

    def forward(self, x):
        return torch.sigmoid(super().forward(x)).squeeze(-1)


# ------------------ #
# helper functions   #
# ------------------ #
def init_recurrent_weights(self):
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)


def lstm_forget_gate_init(lstm_layer):
    for name, parameter in lstm_layer.named_parameters():
        if not "bias" in name: continue
        n = parameter.size(0)
        start, end = n // 4, n // 2
        parameter.data[start:end].fill_(1.)


def clip_grad_norm_hook(x, max_norm=10):
    total_norm = x.norm()
    total_norm = total_norm ** (1 / 2.)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        return x * clip_coef