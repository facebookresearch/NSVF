# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import get_activation_fn
from fairnr.modules.hyper import HyperFC
from fairnr.modules.module_utils import FCLayer


class BackgroundField(nn.Module):
    """
    Background (we assume a uniform color)
    """
    def __init__(self, out_dim=3, bg_color="1.0,1.0,1.0", min_color=-1, stop_grad=False, background_depth=5.0):
        super().__init__()

        if out_dim == 3:  # directly model RGB
            bg_color = [float(b) for b in bg_color.split(',')] if isinstance(bg_color, str) else [bg_color]
            if min_color == -1:
                bg_color = [b * 2 - 1 for b in bg_color]
            if len(bg_color) == 1:
                bg_color = bg_color + bg_color + bg_color
            bg_color = torch.tensor(bg_color)
        else:    
            bg_color = torch.ones(out_dim).uniform_()
            if min_color == -1:
                bg_color = bg_color * 2 - 1
        self.out_dim = out_dim
        self.bg_color = nn.Parameter(bg_color, requires_grad=not stop_grad)
        self.depth = background_depth

    def forward(self, x, **kwargs):
        return self.bg_color.unsqueeze(0).expand(
            *x.size()[:-1], self.out_dim)


class ImplicitField(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False, with_ln=True, skips=None, spec_init=True):
        super().__init__()
        self.skips = skips
        self.net = []

        prev_dim = in_dim
        for i in range(num_layers):
            next_dim = out_dim if i == (num_layers - 1) else hidden_dim
            if (i == (num_layers - 1)) and outmost_linear:
                self.net.append(nn.Linear(prev_dim, next_dim))
            else:
                self.net.append(FCLayer(prev_dim, next_dim, with_ln=with_ln))
            prev_dim = next_dim
            if (self.skips is not None) and (i in self.skips) and (i != (num_layers - 1)):
                prev_dim += in_dim
        
        if num_layers > 0:
            self.net = nn.ModuleList(self.net)
            if spec_init:
                self.net.apply(self.init_weights)

    def forward(self, x):
        y = self.net[0](x)
        for i in range(len(self.net) - 1):
            if (self.skips is not None) and (i in self.skips):
                y = torch.cat((x, y), dim=-1)
            y = self.net[i+1](y)
        return y

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


class HyperImplicitField(nn.Module):

    def __init__(self, hyper_in_dim, in_dim, out_dim, hidden_dim, num_layers, 
                outmost_linear=False):
        super().__init__()

        self.hyper_in_dim = hyper_in_dim
        self.in_dim = in_dim
        self.net = HyperFC(
            hyper_in_dim,
            1, 256, 
            hidden_dim,
            num_layers,
            in_dim,
            out_dim,
            outermost_linear=outmost_linear
        )

    def forward(self, x, c):
        assert (x.size(-1) == self.in_dim) and (c.size(-1) == self.hyper_in_dim)
        if self.nerfpos is not None:
            x = torch.cat([x, self.nerfpos(x)], -1)
        return self.net(c)(x.unsqueeze(0)).squeeze(0)


class SignedDistanceField(ImplicitField):
    """
    Predictor for density or SDF values.
    """
    def __init__(self, in_dim, hidden_dim, num_layers=1, 
                recurrent=False, with_ln=True, spec_init=True):
        super().__init__(in_dim, in_dim, in_dim, num_layers-1, with_ln=with_ln, spec_init=spec_init)
        self.recurrent = recurrent
        if recurrent:
            assert num_layers > 1
            self.hidden_layer = nn.LSTMCell(input_size=in_dim, hidden_size=hidden_dim)
            self.hidden_layer.apply(init_recurrent_weights)
            lstm_forget_gate_init(self.hidden_layer)
        else:
            self.hidden_layer = FCLayer(in_dim, hidden_dim, with_ln) \
                if num_layers > 0 else nn.Identity()
        prev_dim = hidden_dim if num_layers > 0 else in_dim
        self.output_layer = nn.Linear(prev_dim, 1)

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
    def __init__(self, in_dim, hidden_dim, num_layers, 
                with_alpha=False, with_ln=True, spec_init=True):
        out_dim = 3 if not with_alpha else 4
        super().__init__(in_dim, out_dim, hidden_dim, num_layers, 
            outmost_linear=True, with_ln=with_ln, spec_init=spec_init)


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