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
from fairdr.modules.implicit import ImplicitField, PixelRenderer
from fairdr.modules.raymarcher import UniformSearchRayMarcher, SimpleSphereTracer, LSTMSphereTracer
from fairdr.data.geometry import ray


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
        parser.add_argument('--renderer-in-features', type=int, metavar='N',
                            help='renderer hidden dimension for FFN')
        parser.add_argument('--renderer-hidden-dim', type=int, metavar='N',
                            help='renderer hidden dimension for FFN')
        parser.add_argument('--renderer-num-layers', type=int, metavar='N',
                            help='number of FC layers used to renderer')
        parser.add_argument('--raymarching-steps', type=int, metavar='N',
                            help='number of steps for ray-marching')
        parser.add_argument('--lstm-raymarcher', action='store_true',
                            help='model the raymarcher with LSTM cells.')

    def forward(self, ray_start, ray_dir, **kwargs):
        # ray intersection
        depths, _ = self.raymarcher(
            self.field.field, 
            ray_start, ray_dir, 
            steps=self.args.raymarching_steps)

        # return color
        predicts = self.field(ray(ray_start, ray_dir, depths))
        
        # # gradient penalty
        # query_grad = grad()

        return {
            'predicts': predicts,
            'depths': depths
        }

    @torch.no_grad()
    def visualize(self, sample, shape_id=0, view_id=0):
        output = self.forward(
            sample['ray_start'][shape_id:shape_id+1, view_id:view_id+1], 
            sample['ray_dir'][shape_id:shape_id+1, view_id:view_id+1])
        
        def reverse(img, min_val=-1, max_val=1):
            sizes = img.size()
            side_len = int(sizes[0]**0.5)
            if len(sizes) == 1:
                img = img.reshape(side_len, side_len)
            else:
                img = img.reshape(side_len, side_len, -1)
            return ((img - min_val) / (max_val - min_val)).clamp(min=0, max=1).to('cpu')

        #  * (sample['alpha'][shape_id, view_id][:, None])
        images = {
            'depth/{}_{}:HW'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': output['depths'][0, 0], 'min_val': 0.5, 'max_val': 5},
            'rgb/{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': output['predicts'][0, 0]},
            'target/{}_{}:HWC'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': sample['rgb'][shape_id, view_id]},
            'target_depth/{}_{}:HW'.format(
                sample['shape'][shape_id, view_id], sample['view'][shape_id][view_id]):
                {'img': sample['depths'][shape_id, view_id], 'min_val': 0.5, 'max_val': 5},
        }
        images = {
            tag: reverse(**images[tag]) for tag in images
        }
        return images


class SRNField(Field):

    def __init__(self, args):
        super().__init__(args)
        self.field = ImplicitField(args)
        self.renderer = PixelRenderer(args)

    def get_features(self, xyz):
        return self.field(xyz)

    def forward(self, xyz):
        return self.renderer(self.get_features(xyz))


class SRNRaymarcher(Raymarcher):

    def __init__(self, args):
        super().__init__(args)
        if args.lstm_raymarcher:
            self.raymarcher = LSTMSphereTracer(args) 
        else:
            self.raymarcher = SimpleSphereTracer(args)

    def forward(self, feature_fn, ray_start, ray_dir, steps=4):
        return self.raymarcher.search(
            feature_fn,
            ray_start.unsqueeze(-2).expand_as(ray_dir),
            ray_dir, steps=steps, min=0.05)


@register_model_architecture("scene_representation_networks", "srn_base")
def base_architecture(args):
    args.ffn_embed_dim = getattr(args,  "ffn_embed_dim", 256)
    args.ffn_num_layers = getattr(args, "ffn_num_layers", 2)
    args.input_features = getattr(args, "input_features", 3)
    args.output_features = getattr(args, "output_features", 256)
    args.renderer_in_features = args.output_features
    args.renderer_hidden_dim = getattr(args, "renderer_hidden_dim", 256)
    args.renderer_num_layers = getattr(args, "renderer_num_layers", 3)
    args.raymarching_steps = getattr(args, "raymarching_steps", 5)
    args.lstm_raymarcher = getattr(args, "lstm_raymarcher", True)

@register_model_architecture("scene_representation_networks", "srn_simple")
def simple_architecture(args):
    args.lstm_raymarcher = getattr(args, "lstm_raymarcher", False)
    base_architecture(args)