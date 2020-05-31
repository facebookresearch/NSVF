# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.utils import get_activation_fn
from fairdr.modules.utils import FCLayer, ResFCLayer
from fairdr.modules.hyper import HyperFC
from fairdr.modules.linear import NeRFPosEmbLinear

class ImplicitField(nn.Module):
    
    """
    An implicit field is a neural network that outputs a vector given any query point.
    """
    def __init__(self, 
        args, in_dim, out_dim, hidden_dim, num_layers, outmost_linear=False, pos_proj=0):
        super().__init__()
        self.args = args

        if pos_proj > 0:
            new_in_dim = in_dim * 2 * pos_proj
            self.nerfpos = NeRFPosEmbLinear(in_dim, new_in_dim, no_linear=True)
            in_dim = new_in_dim + in_dim
        else:
            self.nerfpos = None

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
        if self.nerfpos is not None:
            x = torch.cat([x, self.nerfpos(x)], -1)
        return self.net(x)


class HyperImplicitField(nn.Module):

    def __init__(self, args, hyper_in_dim, in_dim, out_dim, hidden_dim, num_layers, outmost_linear=False, pos_proj=0):
        super().__init__()

        self.args = args
        if pos_proj > 0:
            new_in_dim = in_dim * 2 * pos_proj
            self.nerfpos = NeRFPosEmbLinear(in_dim, new_in_dim, no_linear=True)
            in_dim = new_in_dim + in_dim
        else:
            self.nerfpos = None

        self.net = HyperFC(
            hyper_in_dim,
            1, 256, 
            hidden_dim,
            num_layers,
            in_dim,
            out_dim,
            outermost_linear=outmost_linear
        )

    def forward(self, x, i, c):
        if self.nerfpos is not None:
            x = torch.cat([x, self.nerfpos(x)], -1)
        net = self.net(c)
        
        def flat2mx(x, i):
            batch_size = c.size(0)
            x_slices = [x[i==b] for b in range(batch_size)]
            max_size = max([xi.size(0) for xi in x_slices])
            new_x = x.new_zeros(batch_size, max_size, x.size(-1))
            for b in range(batch_size):
                new_x[b, :x_slices[b].size(0)] = x_slices[b]
            return new_x

        def mx2flat(x, i):
            batch_size = c.size(0)
            new_x = x.new_zeros(i.size(0), x.size(-1))
            for b in range(batch_size):
                size = (i == b).sum()
                new_x[i == b] = x[b, : size]
            return new_x

        return mx2flat(net(flat2mx(x, i)), i)


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
    def __init__(self, args, in_dim, hidden_dim, num_layers, with_alpha=False):
        out_dim = 3 if not with_alpha else 4
        super().__init__(args, in_dim, out_dim, hidden_dim, num_layers, outmost_linear=True)


class SphereTextureField(TextureField):

    def forward(self, ray_start, ray_dir, min_depth=5.0, steps=10):
        from fairseq import pdb; pdb.set_trace()



class DiffusionSpecularField(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, raydir_dim, num_layers, dropout=0.0):
        super().__init__()
        self.args = args
        self.raydir_dim = raydir_dim

        self.featureField = ImplicitField(args, in_dim, hidden_dim, hidden_dim, num_layers-2, outmost_linear=False)
        self.diffuseField = ImplicitField(args, hidden_dim, 3, hidden_dim, num_layers=1, outmost_linear=True)
        self.specularField = ImplicitField(args, hidden_dim + raydir_dim, 3, hidden_dim, num_layers=1, outmost_linear=True)
        self.dropout = dropout

    def forward(self, x):
        x, r = x[:, :-self.raydir_dim], x[:, -self.raydir_dim:]
        f = self.featureField(x)
        cd = self.diffuseField(f)
        cs = self.specularField(torch.cat([f, r], -1))

        if self.dropout == 0:
            return cd + cs
            
        # BUG: my default rgb is -1 ~ 1
        if self.training and self.dropout > 0:
            cs = cs * (cs.new_ones(cs.size(0)).uniform_() > self.dropout).type_as(cs)[:, None]
        else:
            cs = cs * (1 - self.dropout)

        if getattr(self.args, "min_color", -1) == -1:
            return  (cd + cs) * 2 - 1  # 0 ~ 1 --> -1 ~ 1
        return cd + cs


# bash scripts/generate/generate_lego.sh $MODEL bulldozer6 2 &
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