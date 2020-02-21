# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This is a re-implementation of 
"Scene Representation Networks: 
Continuous 3D-Structure-Aware Neural Scene Representations"
https://vsitzmann.github.io/srns/
"""

import torch
import torch.nn as nn
from torch.autograd import grad

from fairseq.models import (
    register_model,
    register_model_architecture
)

from fairdr.models.fairdr_model import BaseModel, Field, Raymarcher
from fairdr.modules.implicit import ImplicitField, SignedDistanceField, TextureField
from fairdr.modules.raymarcher import IterativeSphereTracer
from fairdr.data.geometry import ray
from fairdr.data.data_utils import recover_image
from fairdr.modules.utils import gradient_bridage


@register_model('scene_representation_networks')
class SRNModel(BaseModel):

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        base_architecture(args)
        field = cls.build_field(args)
        raymarcher = cls.build_raymarcher(args)
        return cls(args, field, raymarcher)
    
    @classmethod
    def build_field(cls, args):
        return SRNField(args)

    @classmethod
    def build_raymarcher(cls, args):
        return SRNRaymarcher(args)

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
        parser.add_argument('--hidden-sdf', type=int, metavar='N', 
                            help='hidden dimension of SDF'),
        parser.add_argument('--lstm-sdf',  action='store_true', 
                            help='model the raymarcher with LSTM cells.')
        parser.add_argument('--hidden-textures', type=int, metavar='N',
                            help='renderer hidden dimension for FFN')
        parser.add_argument('--num-layer-textures', type=int, metavar='N',
                            help='number of FC layers used to renderer')
        parser.add_argument('--raymarching-steps', type=int, metavar='N',
                            help='number of steps for ray-marching')
        parser.add_argument('--jump-to-max-depth', type=float, metavar='D',
                            help='give up ray-marching and directly predict maximum. controlled by gater')
        parser.add_argument('--gradient-penalty', action='store_true')
        parser.add_argument('--implicit-gradient', action='store_true')


    def forward(self, ray_start, ray_dir, **kwargs):
        # ray intersection
        depths, _ = self.raymarcher(
            self.field.get_sdf, 
            ray_start, ray_dir, 
            steps=self.args.raymarching_steps)
        points = ray(ray_start, ray_dir, depths.unsqueeze(-1))

        # return color
        predicts = self.field(points)
        
        # gradient penalty
        if self.args.gradient_penalty and self.training:
            grad_penalty = self.field.get_grad_penalty(points.view(-1, 3), random=True)
        else:
            grad_penalty = 0

        return {
            'predicts': predicts,
            'depths': depths,
            'grad_penalty': grad_penalty
        }

    @torch.no_grad()
    def visualize(self, sample, i=0):
        # only visualize one
        if i > -1:
            _sample = {
                    key: sample[key][0:1, i:i+1]
                    for key in sample
                        if sample[key] is not None and  
                            key in ('ray_start', 'ray_dir', 'view', 'rgb', 'depths')
            }
            _sample['shape'] = sample['shape'][0:1]
            sample = _sample

        output = self.forward(sample['ray_start'], sample['ray_dir'])
        images = {
            'depth/{}_{}:HW'.format(
                sample['shape'][0], sample['view'][0][0]):
                {'img': output['depths'][0, 0], 'min_val': 0.5, 'max_val': 5},
            'rgb/{}_{}:HWC'.format(
                sample['shape'][0], sample['view'][0][0]):
                {'img': output['predicts'][0, 0]},
            'target/{}_{}:HWC'.format(
                sample['shape'][0], sample['view'][0][0]):
                {'img': sample['rgb'][0, 0]}
                if sample.get('rgb', None) is not None else None,
            'target_depth/{}_{}:HW'.format(
                sample['shape'][0], sample['view'][0][0]):
                {'img': sample['depths'][0, 0], 'min_val': 0.5, 'max_val': 5}
                if sample.get('depths', None) is not None else None,
        }
        images = {
            tag: recover_image(**images[tag]) for tag in images if images[tag] is not None
        }
        return images


class SRNField(Field):

    def __init__(self, args):
        super().__init__(args)
        self.feature_field = ImplicitField(
            args, 
            args.input_features, 
            args.output_features, 
            args.hidden_features, 
            args.num_layer_features)
        self.signed_distance_field = SignedDistanceField(
            args,
            args.output_features,
            args.hidden_sdf,
            args.lstm_sdf)
        self.renderer = TextureField(
            args,
            args.output_features,
            args.hidden_textures,
            args.num_layer_textures)

    def get_feature(self, xyz):
        return self.feature_field(xyz)

    def get_sdf(self, xyz, state=None):
        return self.signed_distance_field(self.feature_field(xyz), state)

    def get_texture(self, xyz):
        return self.renderer(self.feature_field(xyz))

    def forward(self, xyz):
        return self.get_texture(xyz)

    def get_grad_penalty(self, xyz, min=-5, max=5, random=True):
        if random:
            xyz = torch.zeros_like(xyz).uniform_(min, max).requires_grad_()
        else:
            xyz = xyz.detach().requires_grad_()

        delta = self.get_sdf(xyz, None)[0]
        gradients = grad(outputs=delta.sum(), inputs=xyz,
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty


class SRNRaymarcher(Raymarcher):

    def __init__(self, args):
        super().__init__(args)
        self.raymarcher = IterativeSphereTracer(args)
        self.implicit = args.implicit_gradient

    def _forward(self, sdf_fn, ray_start, ray_dir, steps=4):
        return self.raymarcher.search(
            sdf_fn,
            ray_start.expand_as(ray_dir),
            ray_dir, steps=steps, min=0.05)

    def forward(self, sdf_fn, ray_start, ray_dir, steps=4):
        if not self.implicit or not self.training:
            return self._forward(sdf_fn, ray_start, ray_dir, steps)
        
        assert not self.args.lstm_sdf, "implicit gradient does not support LSTM."
        with torch.no_grad():
            # forward: search intersection
            depths, states = self._forward(sdf_fn, ray_start, ray_dir, steps)
        
        depths = depths.detach().requires_grad_()
        delta = sdf_fn(ray(ray_start, ray_dir, depths.unsqueeze(-1)), None)[0]
        grad_depth = grad(outputs=delta.sum(), inputs=depths, retain_graph=True)[0]
        depths = gradient_bridage(depths, delta, -1.0 / (grad_depth + 1e-7))

        return depths, states


@register_model_architecture("scene_representation_networks", "srn_base")
def base_architecture(args):
    args.num_layer_features = getattr(args, "num_layer_features", 2)
    args.hidden_features = getattr(args,  "hidden_features", 256)
    args.input_features = getattr(args, "input_features", 3)
    args.output_features = getattr(args, "output_features", 256)
    args.hidden_sdf = getattr(args, "hidden_sdf", 16)
    args.hidden_textures = getattr(args, "hidden_textures", 256)
    args.num_layer_textures = getattr(args, "num_layer_textures", 3)
    args.raymarching_steps = getattr(args, "raymarching_steps", 10)
    args.lstm_sdf = getattr(args, "lstm_sdf", True)
    args.gradient_penalty = getattr(args, "gradient_penalty", False)
    args.implicit_gradient = getattr(args, "implicit_gradient", False)

@register_model_architecture("scene_representation_networks", "srn_simple")
def simple_architecture(args):
    args.lstm_sdf = getattr(args, "lstm_sdf", False)
    base_architecture(args)