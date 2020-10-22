# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy
from torch.autograd import grad
from collections import OrderedDict
from fairnr.modules.implicit import (
    ImplicitField, SignedDistanceField,
    TextureField, HyperImplicitField, BackgroundField
)
from fairnr.modules.linear import NeRFPosEmbLinear
from fairnr.data.geometry import sample_on_sphere

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
    
    def __init__(self, args, no_bg=False):
        super().__init__(args)

        # additional arguments
        self.chunk_size = getattr(args, "chunk_size", 256) * 256
        self.deterministic_step = getattr(args, "deterministic_step", False)       
        
        # background field
        if not no_bg:
            self.min_color = getattr(args, "min_color", -1)
            self.trans_bg = getattr(args, "transparent_background", "1.0,1.0,1.0")
            self.sgbg = getattr(args, "background_stop_gradient", True)
            self.bg_color = BackgroundField(
                bg_color=self.trans_bg, 
                min_color=self.min_color, stop_grad=self.sgbg)       
            
            if getattr(args, "background_network", False):
                new_args = deepcopy(args)
                new_args.__dict__.update({'inputs_to_density': 'pos:10:4', 'inputs_to_texture': 'feat:0:256, ray:4'})
                self.bg_field = RaidanceField(new_args, no_bg=True)
            else:
                self.bg_field = None

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
        parser.add_argument('--background-network', action='store_true')

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

        if (('texture' in outputs) and ("normal" in self.tex_filters)) or ("normal" in outputs):
            assert 'sigma' in inputs, "sigma must be pre-computed"
            assert 'pos' in inputs, "position is used to compute sigma"
            grad_pos, = grad(
                outputs=inputs['sigma'], inputs=inputs['pos'], 
                grad_outputs=torch.ones_like(inputs['sigma']), 
                retain_graph=True)
            inputs['normal'] = F.normalize(-grad_pos, p=2, dim=1)  # BUG: gradient direction reversed.

        if 'texture' in outputs:        
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
        
        # for now we fix the input types
        assert [name for name in self.tex_filters] == ['feat', 'pos', 'normal', 'ray']

        # rebuild the renderer
        self.D = getattr(args, "compressed_light_dim", 64)  # D
        self.renderer = nn.ModuleDict(
            {
                "light-transport": nn.Sequential(
                    ImplicitField(
                    in_dim=sum([self.tex_input_dims[t] for t in [2, 3]]),
                    out_dim=self.D * 3,
                    hidden_dim=args.texture_embed_dim,
                    num_layers=args.texture_layers,
                    outmost_linear=True
                ), nn.Sigmoid()),  # f(v, n, w)
                "lighting": nn.Sequential(
                    ImplicitField(
                    in_dim=sum([self.tex_input_dims[t] for t in [0, 1]]),
                    out_dim=self.D * 3,
                    hidden_dim=args.texture_embed_dim,
                    num_layers=args.texture_layers,
                    outmost_linear=True
                ), nn.ReLU()), # v(x, z, w)
            }
        )
       
    @staticmethod
    def add_args(parser):
        RaidanceField.add_args(parser)
        parser.add_argument('---compressed-light-dim', type=int,
                            help='instead of sampling light directions physically, we compressed the light directions')

    @torch.enable_grad()  # tracking the gradient in case we need to have normal at testing time.
    def forward(self, inputs, outputs=['sigma', 'texture']):
        inputs = super().forward(inputs, outputs=['sigma', 'normal'])
        if 'texture' in outputs:
            lt = self.renderer['light-transport'](
                torch.cat([self.tex_filters['normal'](inputs['normal']),
                           self.tex_filters['ray'](inputs['ray'])], -1)).reshape(-1, self.D, 3)
            li = self.renderer['lighting'](
                torch.cat([self.tex_filters['feat'](inputs['feat']),
                           self.tex_filters['pos'](inputs['pos'])], -1)).reshape(-1, self.D, 3)
            texture = (lt * li).mean(1)
            if self.min_color == -1:
                texture = 2 * texture - 1
            inputs['texture'] = texture
        return inputs


@register_field('disentangled_radiance_field2')
class DisentangledRaidanceField2(RaidanceField):

    def __init__(self, args):
        super().__init__(args)
        
        # for now we fix the input types
        assert [name for name in self.tex_filters] == ['feat', 'pos', 'normal', 'ray']

        # rebuild the renderer
        self.D = getattr(args, "compressed_light_dim", 16)  # D
        self.renderer = nn.ModuleDict(
            {
                "light-transport": ImplicitField(
                    in_dim=sum([self.tex_input_dims[t] for t in [2, 3]]),
                    out_dim=args.texture_embed_dim,
                    hidden_dim=args.texture_embed_dim,
                    num_layers=args.texture_layers,
                    outmost_linear=False
                ),  # f(v, n)
                "lighting": ImplicitField(
                    in_dim=sum([self.tex_input_dims[t] for t in [0, 1]]),
                    out_dim=args.texture_embed_dim,
                    hidden_dim=args.texture_embed_dim,
                    num_layers=args.texture_layers,
                    outmost_linear=False
                ), # t(x, z)
                "lt_out": nn.Sequential(TextureField(
                    in_dim=args.texture_embed_dim + self.tex_input_dims[-1],
                    hidden_dim=128,
                    num_layers=0), nn.Sigmoid()
                ),  # f(v, n, wi)
                "li_out": nn.Sequential(TextureField(
                    in_dim=args.texture_embed_dim + self.tex_input_dims[-1],
                    hidden_dim=128,
                    num_layers=0), nn.ReLU()
                ),  # t(x, n, wi)
            }
        )
       
    @staticmethod
    def add_args(parser):
        RaidanceField.add_args(parser)
        parser.add_argument('--compressed-light-dim', type=int,
                            help='instead of sampling light directions physically, we compressed the light directions')

    @torch.enable_grad()  # tracking the gradient in case we need to have normal at testing time.
    def forward(self, inputs, outputs=['sigma', 'texture']):
        inputs = super().forward(inputs, outputs=['sigma', 'normal'])
        if 'texture' in outputs:
            lt = self.renderer['light-transport'](
                torch.cat([self.tex_filters['normal'](inputs['normal']),
                           self.tex_filters['ray'](inputs['ray'])], -1))
            li = self.renderer['lighting'](
                torch.cat([self.tex_filters['feat'](inputs['feat']),
                           self.tex_filters['pos'](inputs['pos'])], -1))

            def get_chunk_color(D):
                sampled_light_dirs = sample_on_sphere(
                    lt.new_zeros(D, lt.size(0)).uniform_(),
                    li.new_zeros(D, lt.size(0)).uniform_())
                sampled_light_dirs = self.tex_filters['ray'](sampled_light_dirs)
                ct = self.renderer['lt_out'](
                    torch.cat([lt.unsqueeze(0).repeat(D, 1, 1), sampled_light_dirs], -1))
                ci = self.renderer['li_out'](
                    torch.cat([li.unsqueeze(0).repeat(D, 1, 1), sampled_light_dirs], -1))
                return (ct * ci).sum(0)
                
            num_lights = self.D if self.training else 64
            texture, step = 0, 4
            for start in range(0, num_lights, step):
                end = min(start + step, num_lights)
                texture = texture + get_chunk_color(end - start) / num_lights         
            
            if self.min_color == -1:
                texture = 2 * texture - 1
            inputs['texture'] = texture
        return inputs