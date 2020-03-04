# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairdr.modules.utils import FCLayer
from fairdr.data.geometry import ray
from torch.autograd import grad


class UniformSearchRayMarcher(nn.Module):

    """
    Uniform-search requires a value representing occupacy
    It is costly, but esay to parallize if we have GPUs.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def search(self, occupacy_fn, start, ray_dir, min=0.5, max=5, steps=16, tau=0.5):     
        query_arange = min + (max - min) / steps * torch.arange(
            steps + 1, dtype=ray_dir.dtype, device=ray_dir.device)
    
        query_points = start.unsqueeze(-1) + ray_dir.unsqueeze(-1) @ query_arange.unsqueeze(0)
        query_points = query_points.transpose(-1, -2)
        
        query_result = occupacy_fn(query_points) > tau  # check occupacy
        intersection = (query_result.cumsum(-1) == 0).sum(-1)

        missed = intersection == (steps + 1)
        depths = min + (max - min) / steps * intersection.type_as(query_arange)
        random_depths = torch.zeros_like(depths).uniform_(min, max)
        depths = depths.masked_scatter_(missed, random_depths[missed])  # missed replaced with random depth
        
        return depths, missed


class IterativeSphereTracer(nn.Module):

    def __init__(self, args):
        # TODO: fix the auguments
        super().__init__()
        self.args = args
        self.use_raystart = getattr(args, "use_ray_start", False)

    def search(self, signed_distance_fn, start, ray_dir, min=0.05, max=None, steps=4):
        depths = ray_dir.new_ones(*ray_dir.size()[:-1], requires_grad=True) * min
        states, state = [], None
        if self.use_raystart:
            _, state = signed_distance_fn(start, state=state)  # camera as initial state.
        
        for _ in range(steps):
            query = ray(start, ray_dir, depths.unsqueeze(-1))
            delta, state = signed_distance_fn(query, state=state)
            depths = depths + delta
            states.append((query, delta))

        return depths, [q for q in zip(*states)]
