# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairdr.modules.linear import FCLayer
from fairdr.data.geometry import ray


class UniformSearchRayMarcher(nn.Module):

    """
    Uniform-search requires a value representing occupacy
    It is costly, but esay to parallize if we have GPUs.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def search(self, occupacy_fn, start, ray_dir, min=0.5, max=3.5, steps=16, tau=0.5):     
        query_arange = min + (max - min) / steps * torch.arange(
            steps + 1, dtype=ray_dir.dtype, device=ray_dir.device)
        
        query_points = start.unsqueeze(-1) + ray_dir.unsqueeze(-1) @ query_arange.unsqueeze(0)
        query_points = query_points.transpose(-1, -2)
        
        query_result = occupacy_fn(query_points).squeeze(-1) > tau  # check occupacy
        intersection = (query_result.cumsum(-1) == 0).sum(-1)

        missed = intersection == (steps + 1)
        depths = min + (max - min) / steps * intersection.type_as(query_arange)
        random_depths = torch.zeros_like(depths).uniform_(min, max)
        depths = depths.masked_scatter_(missed, random_depths[missed])  # missed replaced with random depth
        
        return depths, missed


class SimpleSphereTracer(nn.Module):

    def __init__(self, args):
        # TODO: fix the auguments
        super().__init__()
        self.args = args
        self.ffn = FCLayer(args.output_features, 16)
        self.signed_distance_field = nn.Linear(16, 1)

    def search(self, feature_fn, start, ray_dir, min=0.05, max=None, steps=4):
        depths = ray_dir.new_ones(*ray_dir.size()[:3]) * min
        states = []
        for _ in range(steps):
            query = ray(start, ray_dir, depths)
            delta = self.signed_distance_field(
                self.ffn(feature_fn(query))).squeeze(-1) 
            depths = depths + delta
            states.append((query, delta))
        return depths, states


class LSTMSphereTracer(SimpleSphereTracer):
    def __init__(self, args):
        super().__init__(args)

        hidden_size = 16
        self.lstm = nn.LSTMCell(input_size=self.args.output_features,
                                hidden_size=16)
        self.lstm.apply(init_recurrent_weights)
        lstm_forget_gate_init(self.lstm)

        self.signed_distance_field = nn.Linear(hidden_size, 1)
        self.counter = 0

    def search(self, feature_fn, start, ray_dir, min=0.05, max=None, steps=4):
        depths = ray_dir.new_ones(*ray_dir.size()[:3]) * min
        states = [None]

        for step in range(steps):
            query = ray(start, ray_dir, depths)
            query_feature = feature_fn(query)
            query_shape = query_feature.size()

            state = self.lstm(query_feature.view(-1, self.args.output_features), states[-1])
            if state[0].requires_grad:
                state[0].register_hook(lambda x: x.clamp(min=-10, max=10))

            delta = self.signed_distance_field(state[0]).view(*query_shape[:-1])
            depths = depths + delta
        return depths, states


# -------
# helper functions
# -------
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
