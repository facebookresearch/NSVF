# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import cv2, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import (
    register_model,
    register_model_architecture
)
from fairdr.modules.pointnet2.pointnet2_utils import (
    aabb_ray_intersect, 
    uniform_ray_sampling
)
from fairdr.data.geometry import ray, trilinear_interp
from fairdr.models.srn_model import SRNModel, base_architecture
from fairdr.models.fairdr_model import Field, Raymarcher
from fairdr.modules.linear import Linear, NeRFPosEmbLinear
from fairdr.modules.implicit import (
    ImplicitField, SignedDistanceField, TextureField, DiffusionSpecularField
)
from fairdr.modules.backbone import BACKBONE_REGISTRY

BG_DEPTH = 5.0
MAX_DEPTH = 10000.0

logger = logging.getLogger(__name__)


@register_model('geo_nerf')
class GEONERFModel(SRNModel):

    @classmethod
    def build_field(cls, args):
        return GEORadianceField(args)

    @classmethod
    def build_raymarcher(cls, args):
        return GEORaymarcher(args)

    @staticmethod
    def add_args(parser):
        # basic SRN hyperparameters
        SRNModel.add_args(parser)
    
        # geometry backbone
        parser.add_argument("--backbone", type=str,
                            help="backbone network, encoding features for the input points")
        for backbone in BACKBONE_REGISTRY:
            BACKBONE_REGISTRY[backbone].add_args(parser)

        # field specific parameters
        parser.add_argument('--voxel-size', type=float, metavar='D',
                            help='voxel size of the input points')
        parser.add_argument("--pos-embed", action='store_true', 
                            help='use positional embedding instead of linear projection')
        parser.add_argument('--use-raydir', action='store_true', 
                            help='if set, use view direction as additional inputs')
        parser.add_argument('--raypos-features', type=int, metavar='N', 
                            help='additional to backbone, additional feature dimensions')
        parser.add_argument('--raydir-features', type=int, metavar='N',
                            help='the number of dimension to encode the ray directions')
        parser.add_argument('--saperate-specular', action='store_true',
                            help='if set, use a different network to predict specular (must provide raydir)')
        parser.add_argument('--specular-dropout', type=float, metavar='D',
                            help='if large than 0, randomly drop specular during training')
        
        # ray-marching parameters
        parser.add_argument('--max-hits', type=int, metavar='N',
                            help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='N',
                            help='ray marching step size')
        parser.add_argument('--discrete-regularization', action='store_true',
                            help='if set, a zero mean unit variance gaussian will be added to encougrage discreteness')
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')

        # training parameters
        parser.add_argument('--chunk-size', type=int, metavar='D', 
                            help='set chunks to go through the network. trade time for memory')
        parser.add_argument('--outer-chunk-size', type=int, metavar='D',
                            help='chunk the input rays in the very beginning if the image was too large.')
        parser.add_argument('--inner-chunking', action='store_true', 
                            help="if set, chunking in the field function, otherwise, chunking in the rays")
        parser.add_argument('--background-stop-gradient', action='store_true')
        
    def _forward(self, ray_start, ray_dir, **kwargs):
        # get geometry features
        feats, xyz = self.field.get_backbone_features()

        # coarse ray-intersection
        S, V, P, _ = ray_dir.size()
        predicts = self.field.bg_color.unsqueeze(0).expand(S * V * P, 3)
        depths = predicts.new_ones(S * V * P) * BG_DEPTH
        missed = predicts.new_ones(S * V * P)
        first_hits = predicts.new_ones(S * V * P).long().fill_(self.field.num_voxels)
        entropy = 0.0

        hits, _ray_start, _ray_dir, state, samples = \
            self.raymarcher.ray_intersection(ray_start, ray_dir, xyz, feats)
        
        # fine-grained raymarching + rendering
        if hits.sum() > 0:   # missed everything
            hits = hits.view(S * V * P).contiguous()
            _predicts, _depths, _missed, entropy = self.raymarcher(
                self.field, _ray_start, _ray_dir, samples, state)
            _predicts = _predicts + self.field.bg_color * _missed.unsqueeze(-1)  # fill background color
            _depths = _depths + BG_DEPTH * _missed

            predicts = predicts.masked_scatter(
                hits.unsqueeze(-1).expand(S * V * P, 3),
                _predicts)
            depths = depths.masked_scatter(
                hits, _depths
            )
            missed = missed.masked_scatter(
                hits, _missed
            )
            first_hits = first_hits.masked_scatter(
                hits, samples[1][:, 0].long()
            )

        # model's output
        return {
            'predicts': predicts.view(S, V, P, 3),
            'depths': depths.view(S, V, P),
            'hits': first_hits.view(S, V, P),
            'missed': missed.view(S, V, P),
            'entropy': entropy,
            'bg_color': self.field.bg_color,
            'other_logs': {
                'voxel_log': self.field.VOXEL_SIZE.item(),
                'marchstep_log': self.raymarcher.MARCH_SIZE.item()}
        }

    @torch.no_grad()
    def adjust(self, action, *args, **kwargs):
        assert self.args.backbone == "dynamic_embedding", \
            "pruning currently only supports dynamic embedding"

        if action == 'prune':
            self.field.pruning(*args, **kwargs)
        
        elif action == 'split':
            self.field.splitting()
            
            # adjust sizes
            self.field.VOXEL_SIZE *= .5
            self.field.MAX_HITS *= 1.5
            self.raymarcher.VOXEL_SIZE *= .5
            self.raymarcher.MAX_HITS *= 1.5

        elif action == 'reduce':
            logger.info("reduce the raymarching step size {} -> {}".format(
                self.field.MARCH_SIZE.item(), self.field.MARCH_SIZE.item() * .5))
            
            # adjust sizes
            self.raymarcher.MARCH_SIZE *= 0.5
            self.field.MARCH_SIZE *= 0.5
        
        else:
            raise NotImplementedError("please specify 'prune, split, shrink' actions")
    
    @property
    def text(self):
        return "GEONERF model (voxel size: {:.4f} ({} voxels), march step: {:.4f})".format(
            self.field.VOXEL_SIZE, self.field.num_voxels, self.raymarcher.MARCH_SIZE)


class GEORadianceField(Field):
    
    def __init__(self, args):
        super().__init__(args)

        # backbone model
        try:
            self.backbone = BACKBONE_REGISTRY[args.backbone](args)
        except Exception:
            raise NotImplementedError("Backbone is not implemented!!!")
        
        # background color
        self.bg_color = nn.Parameter(
            torch.tensor((1.0, 1.0, 1.0)) * getattr(args, "transparent_background", -0.8), 
            requires_grad=(not getattr(args, "background_stop_gradient", False)))

        # arguments
        self.chunk_size = 512 * getattr(args, "chunk_size", 256)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.inner_chunking = getattr(args, "inner_chunking", True)
        self.use_raydir = getattr(args, "use_raydir", False)
        self.raydir_features = getattr(args, "raydir_features", 144)
        self.raypos_features = getattr(args, "raypos_features", 0)

        # build layers
        self.feature_field = ImplicitField(
                args, 
                self.backbone.feature_dim + self.raypos_features, 
                args.output_features, 
                args.hidden_features, 
                args.num_layer_features - 1)
        self.predictor = SignedDistanceField(
                args,
                args.output_features,
                args.hidden_sdf, recurrent=False)
        self.renderer = TextureField(
                args,
                args.output_features + self.raydir_features,
                args.hidden_textures,
                args.num_layer_textures) \
            if not getattr(args, "saperate_specular", False) else DiffusionSpecularField(
                args,
                args.output_features,
                args.hidden_textures,
                self.raydir_features,
                args.num_layer_textures,
                getattr(args, "specular_dropout", 0)
            )

        # [optional] layers
        if self.raypos_features > 0:
            if not getattr(args, "pos_embed", False):
                self.point_proj = Linear(args.input_features, self.raypos_features)
            else:
                self.point_proj = NeRFPosEmbLinear(args.input_features, self.raypos_features, angular=False)

        if self.use_raydir:            
            if not getattr(args, "pos_embed", False):
                self.raydir_proj = Linear(3, self.raydir_features)
            else:
                self.raydir_proj = NeRFPosEmbLinear(3, self.raydir_features, angular=True)

        # register buffers (not learnable)
        self.register_buffer("MAX_HITS", torch.scalar_tensor(args.max_hits))
        self.register_buffer("VOXEL_SIZE", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("MARCH_SIZE", torch.scalar_tensor(args.raymarching_stepsize))

    def get_backbone_features(self, *args, **kwargs):
        return self.backbone(*args, **kwargs)

    def get_feature(self, xyz, point_feats, point_xyz):
        point_feats = self.backbone.get_features(point_feats).view(point_feats.size(0), -1)

        # tri-linear interpolation
        p = ((xyz - point_xyz) / self.VOXEL_SIZE + .5).unsqueeze(1)
        q = (self.backbone.offset.type_as(p) * .5 + .5).unsqueeze(0)
        point_feats = trilinear_interp(p, q, point_feats)
        
        # absolute coordinate features
        if self.raypos_features > 0:
            point_feats = torch.cat([point_feats, self.point_proj(xyz)], -1)
        return self.feature_field(point_feats)

    def _forward(self, xyz, point_feats, point_xyz, dir=None, features=None, outputs=['sigma', 'texture']):
        _data = {}
        if features is None:
            features = self.get_feature(xyz, point_feats, point_xyz)
            _data['features'] = features

        if 'sigma' in outputs:
            sigma = self.predictor(features)[0]
            _data['sigma'] = sigma

        if 'texture' in outputs:
            if dir is not None and self.use_raydir:
                features = torch.cat([features, self.raydir_proj(dir)], -1)
            texture = self.renderer(features)
            _data['texture'] = texture
        return tuple([_data[key] for key in outputs])

    def forward(self, *args, **kwargs):
        if not self.inner_chunking:
            return self._forward(*args, **kwargs)
        
        chunk_size = self.chunk_size
        outputs = zip(*[
                self._forward(*[a[i: i + chunk_size] 
                                if isinstance(a, torch.Tensor) else a 
                                for a in args], **kwargs) 
            for i in range(0, args[-1].shape[0], chunk_size)])
        
        outputs = [torch.cat(o, 0) 
                    if isinstance(o[0], torch.Tensor) 
                    else o for o in outputs]
        return tuple(outputs)

    @torch.no_grad()
    def pruning(self, th=0.5, update=True):    
        feats, xyz = [a[0] for a in self.backbone(None)]
        D = feats.size(-1)
        G = int(self.VOXEL_SIZE / self.MARCH_SIZE)  # how many microgrids to split
        
        logger.info("start pruning")

        # prepare queries for all the voxels
        c = torch.arange(1, 2 * G, 2, device=xyz.device)
        voxel_size = self.VOXEL_SIZE
        ox, oy, oz = torch.meshgrid([c, c, c])
        offsets = (torch.cat([
                    ox.reshape(-1, 1), 
                    oy.reshape(-1, 1), 
                    oz.reshape(-1, 1)], 1).type_as(xyz) - G) / (2 * float(G))
        
        point_xyz = (xyz[:, None, :] + offsets[None, :, :] * voxel_size).reshape(-1, 3)
        point_feats = feats[:, None, :].repeat(1, G ** 3, 1).reshape(-1, D)
        queries = xyz[:, None, :].repeat(1, G ** 3, 1).reshape(-1, 3)

        # query the field
        logger.info("evaluating {}x{}={} micro grids, th={}".format(
            xyz.size(0), G ** 3, queries.size(0), th))
        sigma,  = self.forward(queries, point_feats, point_xyz, outputs=['sigma'])
        sigma_dist = torch.relu(sigma) * self.MARCH_SIZE
        sigma_dist = sigma_dist.reshape(-1, G ** 3)

        alpha = 1 - torch.exp(-sigma_dist.sum(-1))   # probability of filling the full voxel
        keep = (alpha > th)

        if update:
            logger.info("pruning done. before: {}, after: {} voxels".format(xyz.size(0), keep.sum()))
            self.backbone.pruning(keep=keep)
        return keep

    @torch.no_grad()
    def splitting(self):
        logger.info("half the voxel size {} -> {}".format(
            self.VOXEL_SIZE.item(), self.VOXEL_SIZE.item() * .5))
        self.backbone.splitting(self.VOXEL_SIZE * .5)
        logger.info("Total voxel number increases to {}".format(self.num_voxels))
    
    @property
    def num_voxels(self):
        return self.backbone.keep.sum()


class GEORaymarcher(Raymarcher):

    def __init__(self, args):
        super().__init__(args) 
        self.chunk_size = 512 * getattr(args, "chunk_size", 256)
        self.inner_chunking = getattr(args, "inner_chunking", True)
        self.discrete_reg = getattr(args, "discrete_regularization", False)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.expectation = getattr(args, "expectation", "rgb")

        # register buffers (not learnable)
        self.register_buffer("MAX_HITS", torch.scalar_tensor(args.max_hits))
        self.register_buffer("VOXEL_SIZE", torch.scalar_tensor(args.voxel_size))
        self.register_buffer("MARCH_SIZE", torch.scalar_tensor(args.raymarching_stepsize))

    def ray_intersection(self, ray_start, ray_dir, point_xyz, point_feats):
        VOXEL_SIZE, MARCH_SIZE, MAX_HITS = self.VOXEL_SIZE, self.MARCH_SIZE, self.MAX_HITS
        
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3)
        ray_dir = ray_dir.reshape(S, V * P, 3)
        ray_intersect_fn = aabb_ray_intersect
        pts_idx, min_depth, max_depth = ray_intersect_fn(
            VOXEL_SIZE, MAX_HITS, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        pts_idx = (pts_idx + H * torch.arange(S, 
            device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None])
        
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        if hits.sum() <= 0:            # missed everything
            return hits, None, None, None, None
        
        pts_idx = pts_idx[hits]
        min_depth = min_depth[hits]
        max_depth = max_depth[hits]
        max_steps = int(((max_depth - min_depth).sum(-1) / MARCH_SIZE).ceil_().max())
        sampled_idx, sampled_depth, sampled_dists = [p.squeeze(0) for p in 
            uniform_ray_sampling(
                MARCH_SIZE, max_steps, 
                pts_idx.unsqueeze(0), min_depth.unsqueeze(0), max_depth.unsqueeze(0),
                self.deterministic_step or (not self.training)
            )]
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        
        # prepare output
        ray_start = ray_start[hits]
        ray_dir = ray_dir[hits]

        point_feats = point_feats.view(S * H, D)
        point_xyz = point_xyz.view(S * H, 3)

        return hits, ray_start, ray_dir, (point_feats, point_xyz), (sampled_depth, sampled_idx, sampled_dists)

    def _forward(self, field_fn, ray_start, ray_dir, samples, state=None):
        """
        chunks: set > 1 if out-of-memory. it can save some memory by time.
        """
        point_feats, point_xyz = state
        sampled_depth, sampled_idx, sampled_dists = samples
        sampled_idx = sampled_idx.long()

        H, D = point_feats.size()

        # only compute when the ray hits
        sample_mask = sampled_idx.ne(-1)
        if sample_mask.sum() == 0:  # miss everything skip
            return torch.zeros_like(sampled_depth), sampled_depth.new_zeros(*sampled_depth.size(), 3)

        queries = ray(ray_start.unsqueeze(1), ray_dir.unsqueeze(1), sampled_depth.unsqueeze(2))
        queries = queries[sample_mask]
        querie_dirs = ray_dir.unsqueeze(1).expand(*sampled_depth.size(), ray_dir.size()[-1])[sample_mask]
        sampled_idx = sampled_idx[sample_mask]
        sampled_dists = sampled_dists[sample_mask]

        point_xyz = point_xyz.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), 3))
        point_feats = point_feats.gather(0, sampled_idx.unsqueeze(1).expand(sampled_idx.size(0), D))

        sigma, texture = field_fn(queries, point_feats, point_xyz, querie_dirs)
        noise = 0 if not self.discrete_reg and (not self.training) \
            else torch.zeros_like(sigma).normal_()
        dists = self.MARCH_SIZE if self.deterministic_step \
            else sampled_dists        
        sigma_dist = torch.relu(noise + sigma) * dists
        sigma_dist = torch.zeros_like(sampled_depth).masked_scatter(sample_mask, sigma_dist)
        texture = sigma_dist.new_zeros(*sigma_dist.size(), 3).masked_scatter(
            sample_mask.unsqueeze(-1).expand(*sample_mask.size(), 3), texture)

        return sigma_dist, texture

    def forward(self, field_fn, ray_start, ray_dir, samples, state=None):
        sampled_depth, sampled_idx, sampled_dists = samples

        if self.inner_chunking:
            sigma_dist, texture = self._forward(field_fn, ray_start, ray_dir, samples, state)
        else:
            hits = sampled_idx.ne(-1).sum(0)
            sigma_dist, texture = [], []
            size_so_far, start_step = 0, 0
            for i in range(hits.size(0) + 1):
                if (i == hits.size(0)) or (size_so_far + hits[i] > self.chunk_size):
                    _sigma_dist, _texture = self._forward(
                        field_fn, ray_start, ray_dir, (
                            sampled_depth[:, start_step: i], 
                            sampled_idx[:, start_step: i], 
                            sampled_dists[:, start_step: i]), state)
                    
                    sigma_dist += [_sigma_dist]
                    texture += [_texture]
                    start_step, size_so_far = i, 0
                
                if (i < hits.size(0)):
                    size_so_far += hits[i]
            
            sigma_dist = torch.cat(sigma_dist, 1)
            texture = torch.cat(texture, 1)

        # aggregate along the ray
        shifted_sigma_dist = torch.cat([sigma_dist.new_zeros(sampled_depth.size(0), 1), sigma_dist[:, :-1]], dim=-1)  # shift one step
        probs = ((1 - torch.exp(-sigma_dist.float())) * torch.exp(-torch.cumsum(shifted_sigma_dist.float(), dim=-1))).type_as(sigma_dist)
    
        depth = (sampled_depth * probs).sum(-1)
        missed = 1 - probs.sum(-1)
        rgb = (texture * probs.unsqueeze(-1)).sum(-2)
        
        # entropy regularization
        ee = lambda s: (torch.exp(-s.float()) * s.float()).mean().type_as(s)

        return rgb, depth, missed, ee(sigma_dist[sampled_idx.ne(-1)])


def plain_architecture(args):
    # field
    args.pos_embed = getattr(args, "pos_embed", False)
    args.use_raydir = getattr(args, "use_raydir", False)
    args.raydir_features = getattr(args, "raydir_features", 144)
    args.raypos_features = getattr(args, "raypos_features", 0)
    args.saperate_specular = getattr(args, "saperate_specular", False)
    args.specular_dropout = getattr(args, "specular_dropout", 0.0)
    args.voxel_size = getattr(args, "voxel_size", 0.25)

    # raymarcher
    args.max_hits = getattr(args, "max_hits", 32)
    args.raymarching_stepsize = getattr(args, "raymarching_stepsize", 0.01)
    args.background_stop_gradient = getattr(args, "background_stop_gradient", False)
    args.discrete_regularization = getattr(args, "discrete_regularization", False)
    args.deterministic_step = getattr(args, "deterministic_step", False)

    # training
    args.chunk_size = getattr(args, "chunk_size", 100)
    args.outer_chunk_size = getattr(args, "outer_chunk_size", 400 * 400)
    args.inner_chunking = getattr(args, "inner_chunking", False)
    base_architecture(args)

@register_model_architecture("geo_nerf", "geo_nerf")
def geo_base_architecture(args):
    args.backbone = "dynamic_embedding"
    args.quantized_voxel_path = getattr(args, "quantized_voxel_path", None)
    args.quantized_embed_dim = getattr(args, "quantized_embed_dim", 256)
    plain_architecture(args)
