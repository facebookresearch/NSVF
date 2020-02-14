# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a re-implementation of 
"Differentiable Volumetric Rendering: 
Learning Implicit 3D Representations without 3D Supervision"
https://arxiv.org/pdf/1912.07372.pdf
"""

import torch
import torch.nn as nn
from torch.autograd import Function, grad

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairdr.models.fairdr_model import BaseModel, Field, Raymarcher
from fairdr.modules.implicit import ImplicitField
from fairdr.modules.raymarcher import UniformSearchRayMarcher
from fairdr.data.geometry import ray

class ImplicitGraidentBridage(Function):
    
    @staticmethod
    def forward(ctx, depths, occupancy, g_factor):
        ctx.g_factor = g_factor.detach()
        return depths

    @staticmethod
    def backward(ctx, grad_depths):
        grad_occupancy = ctx.g_factor * grad_depths
        return None, grad_occupancy, None


gradient_bridage = ImplicitGraidentBridage.apply


@register_model('diffentiable_volumetric_rendering')
class DVRModel(BaseModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        field = cls.build_field(args)
        raymarcher = cls.build_raymarcher(args)
        return cls(args, field, raymarcher)
    
    @classmethod
    def build_field(cls, args):
        return DVRField(args)

    @classmethod
    def build_raymarcher(cls, args):
        return DVRRaymarcher(args)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--ffn-embed-dim', type=int, metavar='N',
                            help='field input dimension for FFN')
        parser.add_argument('--ffn-hidden-dim', type=int, metavar='N',
                            help='field hidden dimension for FFN')
        parser.add_argument('--input-features', type=int, metavar='N',
                            help='number of features for query')
        parser.add_argument('--output-features', type=int, metavar='N',
                            help='number of features the field returns')
        parser.add_argument('--ffn-num-layers', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--use-residual', action='store_true')
        parser.add_argument('--raymarching-steps', type=int, metavar='N',
                            help='number of steps for ray-marching')

    def forward(self, ray_start, ray_dir, **kwargs):
        # ray intersection
        depths, missed = self.raymarcher(self.field.occupancy, ray_start, ray_dir, steps=64)
        points = ray(ray_start, ray_dir, depths).detach().requires_grad_()
        occupancy = self.field.occupancy(points).squeeze(-1)

        # gradient multiplier
        if self.training:
            g_factor = grad(outputs=occupancy.sum(), inputs=points, retain_graph=True)[0]
            g_factor = -1 / ((g_factor * ray_dir).sum(-1) + 1e-7)
   
            # build a gradient bridge
            depths = gradient_bridage(depths, occupancy, g_factor)

        # return color
        predicts = self.field(ray(ray_start, ray_dir, depths))

        return {
            'occupancy': occupancy,
            'predicts': predicts, 
            'missed': missed,
            'depths': depths
        }
    
    @torch.no_grad()
    def visualize(self, sample, shape_id=0, view_id=0):
        output = self.forward(
            sample['ray_start'][shape_id, view_id], 
            sample['ray_dir'][shape_id, view_id])
        
        def reverse(img):
            sizes = img.size()
            side_len = int(sizes[0]**0.5)
            if len(sizes) == 1:
                img = img.reshape(side_len, side_len)
            else:
                img = img.reshape(side_len, side_len, -1)
            return ((img + 1) / 2).to('cpu')

        images = {
            'occ_{}_{}:HW'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
            output['occupancy'] * (~output['missed']),
            'rgb_{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
            output['predicts'] * (~output['missed'][:, None]),
            'target_{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
            sample['rgb'][shape_id, view_id] * (sample['alpha'][shape_id, view_id][:, None]),
        }
        images = {
            tag: reverse(images[tag]) for tag in images
        }
        return images


class DVRField(Field):

    def __init__(self, args):
        super().__init__(args)
        self.field = ImplicitField(args)  

    def occupancy(self, xyz):
        return torch.sigmoid(self.field(xyz)[1].narrow(-1, 3, 1))  # 0~1

    def forward(self, xyz):
        return self.field(xyz)[1].narrow(-1, 0, 3)    # -1~1


class DVRRaymarcher(Raymarcher):

    def __init__(self, args):
        super().__init__(args)
        self.raymarcher = UniformSearchRayMarcher(args) 

    @torch.no_grad()
    def forward(self, occupancy_fn, ray_start, ray_dir, steps=16):
        return self.raymarcher.search(
            occupancy_fn,
            ray_start.unsqueeze(-2).expand_as(ray_dir),
            ray_dir, 
            min=0.5, max=3.5, steps=steps)


@register_model_architecture("diffentiable_volumetric_rendering", "dvr_base")
def base_architecture(args):
    args.ffn_embed_dim = getattr(args,  "ffn_embed_dim",   256)
    args.ffn_hidden_dim = getattr(args, "ffn_hidden_dim",  256)
    args.ffn_num_layers = getattr(args, "ffn_num_layers",  3)
    args.input_features = getattr(args, "input_features",  3)   # xyz
    args.output_features = getattr(args, "output_features", 4)  # texture (3) + Occupancy(1)
    args.raymarching_steps = getattr(args, "raymarching_steps", 16)
    args.use_residual = getattr(args, "use_residual", True)