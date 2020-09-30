# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from collections import OrderedDict
from fairnr.modules.implicit import (
    ImplicitField, SignedDistanceField,
    TextureField, HyperImplicitField, BackgroundField
)
from fairnr.modules.linear import NeRFPosEmbLinear

FIELD_REGISTRY = {}

def register_field(name):
    def register_field_cls(cls):
        if name in FIELD_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        FIELD_REGISTRY[name] = cls
        return cls
    return register_field_cls


def get_field(name):
    if name not in FIELD_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return FIELD_REGISTRY[name]
    

@register_field('abstract_field')
class Field(nn.Module):
    """
    Abstract class for implicit functions
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        pass


@register_field('radiance_field')
class RaidanceField(Field):
    
    def __init__(self, args):
        super().__init__(args)

        # additional arguments
        self.chunk_size = getattr(args, "chunk_size", 256) * 256
        self.deterministic_step = getattr(args, "deterministic_step", False)       
        
        # background field
        self.bg_color = BackgroundField(
            bg_color=getattr(args, "transparent_background", "1.0,1.0,1.0"),
            min_color=getattr(args, "min_color", -1), 
            stop_grad=getattr(args, "background_stop_gradient", False))       
        self.den_filters, self.den_ori_dims, self.den_input_dims = self.parse_inputs(args.inputs_to_density)
        self.tex_filters, self.tex_ori_dims, self.tex_input_dims = self.parse_inputs(args.inputs_to_texture)
        self.den_filters, self.tex_filters = nn.ModuleDict(self.den_filters), nn.ModuleDict(self.tex_filters)
        den_input_dim, tex_input_dim = sum(self.den_input_dims), sum(self.tex_input_dims)
        den_feat_dim = self.tex_input_dims[0]

        # build networks
        if not getattr(args, "hypernetwork", False):
            self.feature_field = ImplicitField(den_input_dim, den_feat_dim, 
                args.feature_embed_dim, args.feature_layers)
        else:
            den_contxt_dim = self.den_input_dims[-1]
            self.feature_field = HyperImplicitField(den_contxt_dim, den_input_dim - den_contxt_dim, 
                den_feat_dim, args.feature_embed_dim, args.feature_layers)
        self.predictor = SignedDistanceField(den_feat_dim, args.density_embed_dim, recurrent=False)
        self.renderer = TextureField(tex_input_dim, args.texture_embed_dim, args.texture_layers) 

    def parse_inputs(self, arguments):
        def fillup(p):
            assert len(p) > 0
            default = 'b' if (p[0] != 'ray') and (p[0] != 'normal') else 'a'

            if len(p) == 1:
                return [p[0], 0, 3, default]
            elif len(p) == 2:
                return [p[0], int(p[1]), 3, default]
            elif len(p) == 3:
                return [p[0], int(p[1]), int(p[2]), default]
            return [p[0], int(p[1]), int(p[2]), p[4]]

        filters, input_dims, output_dims = OrderedDict(), [], []
        for p in arguments.split(','):
            name, pos_dim, base_dim, pos_type = fillup([a.strip() for a in p.strip().split(':')])
            
            if pos_dim > 0:  # use positional embedding
                func = NeRFPosEmbLinear(
                    base_dim, base_dim * pos_dim * 2, 
                    angular=(pos_type == 'a'), 
                    no_linear=True,
                    cat_input=(pos_type != 'a'))
                odim = func.out_dim + func.in_dim if func.cat_input else func.out_dim

            else:
                func = nn.Identity()
                odim = base_dim

            input_dims += [base_dim]
            output_dims += [odim]
            filters[name] = func
        return filters, input_dims, output_dims

    @staticmethod
    def add_args(parser):
        parser.add_argument('--inputs-to-density', type=str,
                            help="""
                                Types of inputs to predict the density.
                                Choices of types are emb or pos. 
                                  use first . to assign sinsudoal frequency.
                                  use second : to assign the input dimension (in default 3).
                                  use third : to set the type -> basic, angular or gaussian
                                Size must match
                                e.g.  --inputs-to-density emb:6:32,pos:4
                                """)
        parser.add_argument('--inputs-to-texture', type=str,
                            help="""
                                Types of inputs to predict the texture.
                                Choices of types are feat, emb, ray, pos or normal.
                                """)

        parser.add_argument('--feature-embed-dim', type=int, metavar='N',
                            help='field hidden dimension for FFN')
        parser.add_argument('--density-embed-dim', type=int, metavar='N', 
                            help='hidden dimension of density prediction'),
        parser.add_argument('--texture-embed-dim', type=int, metavar='N',
                            help='hidden dimension of texture prediction')

        parser.add_argument('--input-embed-dim', type=int, metavar='N',
                            help='number of features for query (in default 3, xyz)')
        parser.add_argument('--output-embed-dim', type=int, metavar='N',
                            help='number of features the field returns')
        parser.add_argument('--raydir-embed-dim', type=int, metavar='N',
                            help='the number of dimension to encode the ray directions')
        parser.add_argument('--disable-raydir', action='store_true', 
                            help='if set, not use view direction as additional inputs')
        parser.add_argument('--add-pos-embed', type=int, metavar='N', 
                            help='using periodic activation augmentation')
        parser.add_argument('--feature-layers', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--texture-layers', type=int, metavar='N',
                            help='number of FC layers used to predict colors')        

        # specific parameters (hypernetwork does not work right now)
        parser.add_argument('--hypernetwork', action='store_true', 
                            help='use hypernetwork to model feature')
        parser.add_argument('--hyper-feature-embed-dim', type=int, metavar='N',
                            help='feature dimension used to predict the hypernetwork. consistent with context embedding')

        # backgound parameters
        parser.add_argument('--background-depth', type=float,
                            help='the depth of background. used for depth visualization')
        parser.add_argument('--background-stop-gradient', action='store_true',
                            help='do not optimize the background color')

    @torch.enable_grad()  # tracking the gradient in case we need to have normal at testing time.
    def forward(self, inputs, outputs=['sigma', 'texture']):
        filtered_inputs, context = [], None
        if 'feat' not in inputs:        
            for i, name in enumerate(self.den_filters):
                d_in, func = self.den_ori_dims[i], self.den_filters[name]
                assert (name in inputs), "the encoder must contain target inputs"
                assert inputs[name].size(-1) == d_in, "{} dimension must match {} v.s. {}".format(
                    name, inputs[name].size(-1), d_in)
                if name == 'context':
                    assert (i == (len(self.den_filters) - 1)), "we force context as the last input"        
                    assert inputs[name].size(0) == 1, "context is object level"
                    context = func(inputs[name])
                else:
                    filtered_inputs += [func(inputs[name])]
            
            filtered_inputs = torch.cat(filtered_inputs, -1)
            if context is not None:
                if getattr(self.args, "hypernetwork", False):
                    filtered_inputs = (filtered_inputs, context)
                else:
                    filtered_inputs = (torch.cat([filtered_inputs, context.repeat(filtered_inputs.size(0), 1)], -1),)
            else:
                filtered_inputs = (filtered_inputs, )
            inputs['feat'] = self.feature_field(*filtered_inputs)
        
        if 'sigma' in outputs:
            assert 'feat' in inputs, "feature must be pre-computed"
            inputs['sigma'] = self.predictor(inputs['feat'])[0]

        if 'texture' in outputs:
            if "normal" in self.tex_filters:
                assert 'sigma' in inputs, "sigma must be pre-computed"
                assert 'pos' in inputs, "position is used to compute sigma"
                grad_pos, = grad(
                    outputs=inputs['sigma'], inputs=inputs['pos'], 
                    grad_outputs=torch.ones_like(inputs['sigma']), 
                    retain_graph=True)
                inputs['normal'] = F.normalize(-grad_pos, p=2, dim=1)  # BUG: gradient direction reversed.
                
            filtered_inputs = []
            for i, name in enumerate(self.tex_filters):
                d_in, func = self.tex_ori_dims[i], self.tex_filters[name]
                assert (name in inputs), "the encoder must contain target inputs"
                assert inputs[name].size(-1) == d_in, "dimension must match"

                filtered_inputs += [func(inputs[name])]
                
            filtered_inputs = torch.cat(filtered_inputs, -1)
            inputs['texture'] = self.renderer(filtered_inputs)

        return inputs



@register_field('disentangled_radiance_field')
class DisentangledRaidanceField(RaidanceField):

    def __init__(self, args):
        super().__init__(args)

        # rebuild the renderer
        self.projected_dim = getattr(args, "projected_dim", 32)  # D
        self.renderer = nn.ModuleDict(
            {
                "light-transport": ImplicitField(args, ),
                "visibility": ImplicitField(args, ),
                "lighting": BackgroundField(args)
            }
        )