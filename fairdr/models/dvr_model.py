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

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairdr.models.fairdr_model import BaseModel, Field, Raymarcher
from fairdr.modules.implicit import ImplicitField, OccupancyField, TextureField
from fairdr.modules.raymarcher import UniformSearchRayMarcher
from fairdr.modules.utils import gradient_bridage
from fairdr.data.geometry import ray
from fairdr.data.data_utils import recover_image


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
        parser.add_argument('--hidden-features', type=int, metavar='N',
                            help='field hidden dimension for FFN')
        parser.add_argument('--input-features', type=int, metavar='N',
                            help='number of features for query')
        parser.add_argument('--output-features', type=int, metavar='N',
                            help='number of features the field returns')
        parser.add_argument('--num-layer-features', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--hidden-occupancy', type=int, metavar='N', 
                            help='hidden dimension of SDF'),
        parser.add_argument('--hidden-textures', type=int, metavar='N',
                            help='renderer hidden dimension for FFN')
        parser.add_argument('--num-layer-textures', type=int, metavar='N',
                            help='number of FC layers used to renderer')
        parser.add_argument('--raymarching-steps', type=int, metavar='N',
                            help='number of steps for ray-marching')

    def forward(self, ray_start, ray_dir, **kwargs):
        # ray intersection
        depths, missed = self.raymarcher(
            self.field.get_occupancy, 
            ray_start, ray_dir, 
            steps=self.args.raymarching_steps)
        
        # get final query points
        points = ray(ray_start, ray_dir, depths).detach().requires_grad_()
        occupancy = self.field.get_occupancy(points)

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
            sample['ray_start'][shape_id:shape_id+1, view_id:view_id+1], 
            sample['ray_dir'][shape_id:shape_id+1, view_id:view_id+1])
        
        images = {
            'depth/{}_{}:HW'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': output['depths'][0, 0], 'min_val': 0.5, 'max_val': 5},
            'rgb/{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': output['predicts'][0, 0] * (~output['missed'][0, 0].unsqueeze(-1))},
            'target/{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': sample['rgb'][shape_id, view_id]},
            'target_depth/{}_{}:HW'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': sample['depths'][shape_id, view_id], 'min_val': 0.5, 'max_val': 5}
                if sample['depths'] is not None else None,
        }
        images = {
            tag: recover_image(**images[tag]) for tag in images if images[tag] is not None
        }
        return images


class DVRField(Field):

    def __init__(self, args):
        super().__init__(args)
        self.feature_field = ImplicitField(
            args, 
            args.input_features, 
            args.output_features, 
            args.hidden_features, 
            args.num_layer_features)
        self.occupancy_field = OccupancyField(
            args,
            args.output_features,
            args.hidden_occupancy, 1)
        self.renderer = TextureField(
            args,
            args.output_features,
            args.hidden_textures,
            args.num_layer_textures)

    def get_occupancy(self, xyz):
        return self.occupancy_field(self.feature_field(xyz))

    def get_texture(self, xyz):
        return self.renderer(self.feature_field(xyz))

    def forward(self, xyz):
        return self.get_texture(xyz)


class DVRRaymarcher(Raymarcher):

    def __init__(self, args):
        super().__init__(args)
        self.raymarcher = UniformSearchRayMarcher(args) 

    @torch.no_grad()
    def forward(self, occupancy_fn, ray_start, ray_dir, steps=16):
        return self.raymarcher.search(
            occupancy_fn,
            ray_start.unsqueeze(-2).expand_as(ray_dir),
            ray_dir, min=0.5, max=5, steps=steps)


@register_model_architecture("diffentiable_volumetric_rendering", "dvr_base")
def base_architecture(args):
    args.num_layer_features = getattr(args, "num_layer_features", 2)
    args.hidden_features = getattr(args,  "hidden_features", 256)
    args.input_features = getattr(args, "input_features", 3)
    args.output_features = getattr(args, "output_features", 256)
    args.hidden_occupancy = getattr(args, "hidden_occupancy", 16)
    args.hidden_textures = getattr(args, "hidden_textures", 256)
    args.num_layer_textures = getattr(args, "num_layer_textures", 3)
    args.raymarching_steps = getattr(args, "raymarching_steps", 32)