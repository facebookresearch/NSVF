# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import grad
from collections import OrderedDict
from fairnr.modules.implicit import (
    ImplicitField, SignedDistanceField,
    TextureField, HyperImplicitField, BackgroundField
)
from fairnr.modules.module_utils import NeRFPosEmbLinear

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
        self.updates = -1

    def forward(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def add_args(parser):
        pass
    
    def set_num_updates(self, num_updates):
        self.updates = num_updates

@register_field('radiance_field')
class RaidanceField(Field):
    
    def __init__(self, args):
        super().__init__(args)

        # additional arguments
        self.chunk_size = getattr(args, "chunk_size", 256) * 256
        self.deterministic_step = getattr(args, "deterministic_step", False)       
        
        # background field
        self.min_color = getattr(args, "min_color", -1)
        self.trans_bg = getattr(args, "transparent_background", "1.0,1.0,1.0")
        self.sgbg = getattr(args, "background_stop_gradient", False)
        self.bg_color = BackgroundField(bg_color=self.trans_bg, min_color=self.min_color, stop_grad=self.sgbg)

        # MLP specs
        self.nerf_style = getattr(args, "nerf_style_mlp", False)  # NeRF style MLPs
        self.with_ln = not getattr(args, "no_layernorm_mlp", False)
        self.skips = getattr(args, "feature_field_skip_connect", None) 
        self.skips = [self.skips] if self.skips is not None else None

        # input specs
        self.den_filters, self.den_ori_dims, self.den_input_dims = self.parse_inputs(args.inputs_to_density)
        self.tex_filters, self.tex_ori_dims, self.tex_input_dims = self.parse_inputs(args.inputs_to_texture)
        self.den_filters, self.tex_filters = nn.ModuleDict(self.den_filters), nn.ModuleDict(self.tex_filters)
        
        # build networks
        self.build_feature_field(args)
        self.build_density_predictor(args)
        self.build_texture_renderer(args)

        if getattr(args, "zero_z_steps", 0) > 0:
            self.register_buffer("zero_z", torch.scalar_tensor(1))  # it will be saved to checkpoint
        else:
            self.zero_z = 0

    def set_num_updates(self, updates):
        self.updates = updates
        if getattr(self.args, "zero_z_steps", 0) <= self.updates:
            self.zero_z = self.zero_z * 0

    def build_feature_field(self, args): 
        den_feat_dim = self.tex_input_dims[0]
        den_input_dim, tex_input_dim = sum(self.den_input_dims), sum(self.tex_input_dims)

        if not getattr(args, "hypernetwork", False):
            self.feature_field = ImplicitField(
                den_input_dim, den_feat_dim, args.feature_embed_dim, 
                args.feature_layers + 2 if not self.nerf_style else 8,          # +2 is to adapt to old code
                with_ln=self.with_ln if not self.nerf_style else False, 
                skips=self.skips if not self.nerf_style else [4],
                spec_init=True if not self.nerf_style else False)  
        else:
            assert (not self.nerf_style), "Hypernetwork does not support NeRF style MLPs yet."
            den_contxt_dim = self.den_input_dims[-1]
            self.feature_field = HyperImplicitField(
                den_contxt_dim, den_input_dim - den_contxt_dim, 
                den_feat_dim, args.feature_embed_dim, args.feature_layers + 2)  # +2 is to adapt to old code
        
    def build_density_predictor(self, args):
        den_feat_dim = self.tex_input_dims[0]
        self.predictor = SignedDistanceField(
            den_feat_dim, args.density_embed_dim, recurrent=False, num_layers=1, 
            with_ln=self.with_ln if not self.nerf_style else False,
            spec_init=True if not self.nerf_style else False)

    def build_texture_renderer(self, args):
        tex_input_dim = sum(self.tex_input_dims)
        self.renderer = TextureField(
            tex_input_dim, args.texture_embed_dim, 
            args.texture_layers + 2 if not self.nerf_style else 2, 
            with_ln=self.with_ln if not self.nerf_style else False,
            spec_init=True if not self.nerf_style else False)

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
            return [p[0], int(p[1]), int(p[2]), p[3]]

        filters, input_dims, output_dims = OrderedDict(), [], []
        for p in arguments.split(','):
            name, pos_dim, base_dim, pos_type = fillup([a.strip() for a in p.strip().split(':')])
            
            if pos_dim > 0:  # use positional embedding
                func = NeRFPosEmbLinear(
                    base_dim, base_dim * pos_dim * 2, 
                    angular=(pos_type == 'a'), 
                    no_linear=True,
                    cat_input=(pos_type == 'b'))
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

        parser.add_argument('--nerf-style-mlp', action='store_true',
                            help='use NeRF style MLPs for implicit function (with skip-connection).')
        parser.add_argument('--no-layernorm-mlp', action='store_true',
                            help='do not use layernorm in MLPs.')
        parser.add_argument('--feature-field-skip-connect', type=int,
                            help='add skip-connection in the feature field.')

        parser.add_argument('--feature-embed-dim', type=int, metavar='N',
                            help='field hidden dimension for FFN')
        parser.add_argument('--density-embed-dim', type=int, metavar='N', 
                            help='hidden dimension of density prediction'),
        parser.add_argument('--texture-embed-dim', type=int, metavar='N',
                            help='hidden dimension of texture prediction')
        parser.add_argument('--feature-layers', type=int, metavar='N',
                            help='number of FC layers used to encode')
        parser.add_argument('--texture-layers', type=int, metavar='N',
                            help='number of FC layers used to predict colors')        
        parser.add_argument('--no-normalize-normal', action='store_true',
                            help='if set, do not normalize the gradient of density as the normal direction.')
        parser.add_argument('--zero-z-steps', type=int, default=0)

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
        if inputs.get('feat', None) is None:        
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
                    filtered_inputs = (torch.cat([filtered_inputs, context.expand(filtered_inputs.size(0), context.size(1))], -1),)
            else:
                filtered_inputs = (filtered_inputs, )
            inputs['feat'] = self.feature_field(*filtered_inputs)
            
        if 'sigma' in outputs:
            assert 'feat' in inputs, "feature must be pre-computed"
            inputs['sigma'] = self.predictor(inputs['feat'])[0]
            
        if ('normal' not in inputs) and (
            (('texture' in outputs) and ("normal" in self.tex_filters)) 
            or ("normal" in outputs)):
            
            assert 'sigma' in inputs, "sigma must be pre-computed"
            assert 'pos' in inputs, "position is used to compute sigma"
            grad_pos, = grad(
                outputs=inputs['sigma'], inputs=inputs['pos'], 
                grad_outputs=torch.ones_like(inputs['sigma'], requires_grad=False), 
                retain_graph=True, create_graph=True)
            if not getattr(self.args, "no_normalize_normal", False):
                inputs['normal'] = F.normalize(-grad_pos, p=2, dim=1)  # BUG: gradient direction reversed.
            else:
                inputs['normal'] = -grad_pos  # no normalization. magnitude also has information?

        if 'texture' in outputs:        
            filtered_inputs = []
            if self.zero_z == 1:
                inputs['feat'] = inputs['feat'] * 0.0  # zero-out latent feature
            inputs['feat_n2'] = (inputs['feat'] ** 2).sum(-1)

            for i, name in enumerate(self.tex_filters):
                d_in, func = self.tex_ori_dims[i], self.tex_filters[name]
                assert (name in inputs), "the encoder must contain target inputs"
                filtered_inputs += [func(inputs[name])] if name != 'sigma' else [func(inputs[name].unsqueeze(-1))]
                
            filtered_inputs = torch.cat(filtered_inputs, -1)
            inputs['texture'] = self.renderer(filtered_inputs)
            
            if self.min_color == 0:
                inputs['texture'] = torch.sigmoid(inputs['texture'])
            
        return inputs


@register_field('sdf_radiance_field')
class SDFRaidanceField(RaidanceField):

    @staticmethod
    def add_args(parser):
        parser.add_argument('--reg-z', action='store_true', help='regularize latent feature')
        parser.add_argument('--dropout-z', type=float, default=0.0)
        parser.add_argument('--project-to-surface', action='store_true', 
            help='project any point to the surface to obtain color.')
        RaidanceField.add_args(parser)

    def build_feature_field(self, args):
        den_feat_dim = self.tex_input_dims[0]
        den_input_dim, tex_input_dim = sum(self.den_input_dims), sum(self.tex_input_dims)

        assert not getattr(args, "hypernetwork", False), "does not support hypernetwork for now"
        assert (den_input_dim == 3) or (
            self.den_filters['pos'].cat_input and len(self.den_filters) == 1), "cat pos in the end"

        num_layers = args.feature_layers + 2 if not self.nerf_style else 8
        skips = self.skips if not self.nerf_style else [4]

        self.feature_field = ImplicitField(
            den_input_dim, den_feat_dim + 1, args.feature_embed_dim,  # +1 is for SDF values
            num_layers, with_ln=False, skips=skips, outmost_linear=True, spec_init=False)  
        
        if getattr(args, "dropout_z", 0.0) > 0.0:
            self.dropout_z = nn.Dropout(p=self.args.dropout_z)
        else:
            self.dropout_z = None

        """ 
        Geometric initialization from https://arxiv.org/pdf/1911.10414.pdf
        This enforce a model to approximate a SDF function: 
            f(x; \theta) \approx |x| - 1   
        """
        bias = 1.0
        for l in range(num_layers):
            lin = self.feature_field.net[l] 
            if l < num_layers - 1:
                lin = lin.net[0]

            if l == num_layers - 1:  # last layer
                torch.nn.init.normal_(lin.weight, mean=math.sqrt(math.pi) / math.sqrt(lin.weight.size(1)), std=0.0001)
                torch.nn.init.constant_(lin.bias, -bias)
            elif l == 0:
                torch.nn.init.constant_(lin.bias, 0.0)
                if den_input_dim > 3:
                    torch.nn.init.constant_(lin.weight[:, :-3], 0.0)
                torch.nn.init.normal_(lin.weight[:, -3:], 0.0, math.sqrt(2) / math.sqrt(lin.weight.size(0)))
            elif (l - 1) in skips:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(lin.weight.size(0)))
                torch.nn.init.constant_(lin.weight[:, :den_input_dim-3], 0.0)
            else:
                torch.nn.init.constant_(lin.bias, 0.0)
                torch.nn.init.normal_(lin.weight, 0.0, math.sqrt(2) / math.sqrt(lin.weight.size(0)))
        
        # force the initial fearures to 0
        self.feature_field.net[7].weight.data[1:] = self.feature_field.net[7].weight.data[1:] * 0.0
        self.feature_field.net[7].bias.data[1:] = self.feature_field.net[7].bias.data[1:] * 0.0
        
    def build_density_predictor(self, args):
        class Sdf2Densityv1(nn.Module):
            def __init__(self):
                super().__init__()
                self.alpha = nn.Parameter(torch.scalar_tensor(10.0), requires_grad=True)
                self.sigma = nn.Parameter(torch.scalar_tensor(50.0), requires_grad=True)
            
            def forward(self, x):
                return self.sigma * torch.tanh(-torch.abs(self.alpha) * x[:, 0]), None

        class Sdf2Densityv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.sigma = nn.Parameter(torch.scalar_tensor(50.0), requires_grad=True)
                # self.alpha = nn.Parameter(torch.scalar_tensor(0.05), requires_grad=True)

            def forward(self, x):
                return -self.sigma * x[:, 0], None

        class Sdf2Densityv3(nn.Module):
            def __init__(self):
                super().__init__()
                self.sigma = nn.Parameter(torch.scalar_tensor(100.0), requires_grad=True)
                # self.alpha = nn.Parameter(torch.scalar_tensor(0.05), requires_grad=True)

            def forward(self, x):
                return F.elu(-self.sigma * x[:, 0]), None

        self.predictor = Sdf2Densityv1()

    @torch.enable_grad()
    def forward(self, inputs, outputs=['sigma', 'texture']):
        if ('sigma' in outputs):
            inputs = super().forward(inputs, ['sigma'])
            inputs['sdf'] = inputs['feat'][:, 0]
            inputs['feat'] = inputs['feat'][:, 1:]  # remove sdf from feature
            if (getattr(self.args, "zero_z_steps", 0) > self.updates) and self.training:
                inputs['feat'] = inputs['feat'] * 0.0  # zero-out latent feature

            if self.dropout_z is not None:
                inputs['feat'] = self.dropout_z(inputs['feat'])  # apply dropout on the feature.
            
        if ('texture' in outputs) or ('normal' in outputs):
            # compute gradient for sdf, no need to normalize them
            inputs['normal'] = grad(
                outputs=inputs['sdf'], inputs=inputs['pos'], 
                grad_outputs=torch.ones_like(inputs['sdf'], requires_grad=False), 
                retain_graph=True, create_graph=True)[0]

            # compute color for points projected on the surface
            if getattr(self.args, "project_to_surface", False):
                inputs['pos'] = inputs['pos'] - inputs['sdf'][:, None] * inputs['normal']
                inputs['feat'] = None
                inputs = super().forward(inputs, outputs=['feat'])
                inputs['feat'] = inputs['feat'][:, 1:]
            
            inputs['feat_n2'] = (inputs['feat'] ** 2).sum(-1)

        if 'texture' in outputs:
            inputs = super().forward(inputs, ['texture'])
        return inputs


@register_field('disentangled_radiance_field')
class DisentangledRaidanceField(RaidanceField):

    def __init__(self, args):
        super().__init__(args)
        
        # for now we fix the input types
        assert [name for name in self.tex_filters][:4] == ['feat', 'pos', 'normal', 'ray']
        lt_in_dim = self.tex_input_dims[2] + self.tex_input_dims[3]
        lg_in_dim = self.tex_input_dims[0] + self.tex_input_dims[1]
        if len(self.tex_filters) > 4:
            lt_in_dim += sum(self.tex_input_dims[4:])
            lg_in_dim += sum(self.tex_input_dims[4:])

        # rebuild the renderer
        self.D = getattr(args, "compressed_light_dim", 64)  # D
        self.renderer = nn.ModuleDict(
            {
                "light-transport": nn.Sequential(
                    ImplicitField(
                    in_dim=lt_in_dim,
                    out_dim=self.D * 3,
                    hidden_dim=args.texture_embed_dim,
                    num_layers=args.texture_layers,
                    outmost_linear=True
                ), nn.Sigmoid()),  # f(v, n, w)
                "lighting": nn.Sequential(
                    ImplicitField(
                    in_dim=lg_in_dim,
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
        h_g, h_brdf, h_l = None, None, None
        if inputs.get('context', None) is not None:
            h_g, h_brdf, h_l = [inputs['context'][k:k+1] for k in range(3)]
            inputs['context'] = h_g
        inputs = super().forward(inputs, outputs=['sigma', 'normal'])
        if 'texture' in outputs:
            lt_inputs = [self.tex_filters['normal'](inputs['normal']), self.tex_filters['ray'](inputs['ray'])]
            if h_brdf is not None:
                lt_inputs += [self.tex_filters['context'](h_brdf).expand(lt_inputs[0].size(0), -1)]
            
            li_inputs = [self.tex_filters['feat'](inputs['feat']), self.tex_filters['pos'](inputs['pos'])]
            if h_l is not None:
                li_inputs += [self.tex_filters['context'](h_l).expand(li_inputs[0].size(0), -1)]
            
            lt = self.renderer['light-transport'](torch.cat(lt_inputs, -1)).reshape(-1, self.D, 3)
            li = self.renderer['lighting'](torch.cat(li_inputs, -1)).reshape(-1, self.D, 3)
            texture = (lt * li).mean(1)
            if self.min_color == -1:
                texture = 2 * texture - 1
            inputs['texture'] = texture
        return inputs