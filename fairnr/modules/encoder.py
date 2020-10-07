# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys
import os
import math
import logging
logger = logging.getLogger(__name__)

from pathlib import Path
from fairnr.data.data_utils import load_matrix
from fairnr.data.geometry import (
    trilinear_interp, splitting_points, offset_points,
    get_edge, build_easy_octree
)
from fairnr.clib import (
    aabb_ray_intersect, triangle_ray_intersect,
    uniform_ray_sampling, svo_ray_intersect
)
from fairnr.modules.linear import FCBlock, Linear, Embedding

MAX_DEPTH = 10000.0
ENCODER_REGISTRY = {}

def register_encoder(name):
    def register_encoder_cls(cls):
        if name in ENCODER_REGISTRY:
            raise ValueError('Cannot register duplicate module ({})'.format(name))
        ENCODER_REGISTRY[name] = cls
        return cls
    return register_encoder_cls


def get_encoder(name):
    if name not in ENCODER_REGISTRY:
        raise ValueError('Cannot find module {}'.format(name))
    return ENCODER_REGISTRY[name]


@register_encoder('abstract_encoder')
class Encoder(nn.Module):
    """
    backbone network
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, **kwargs):
        raise NotImplementedError
    
    @staticmethod
    def add_args(parser):
        pass


@register_encoder('sparsevoxel_encoder')
class SparseVoxelEncoder(Encoder):

    def __init__(self, args, voxel_path=None, bbox_path=None, shared_values=None):
        super().__init__(args)
        self.voxel_path = voxel_path if voxel_path is not None else args.voxel_path
        self.bbox_path = bbox_path if bbox_path is not None else getattr(args, "initial_boundingbox", None)
        assert (self.bbox_path is not None) or (self.voxel_path is not None), \
            "at least initial bounding box or pretrained voxel files are required."
        
        self.voxel_index = None
        if self.voxel_path is not None:
            assert os.path.exists(self.voxel_path), "voxel file must exist"
            assert getattr(args, "voxel_size", None) is not None, "final voxel size is essential."
            
            voxel_size = args.voxel_size

            if Path(self.voxel_path).suffix == '.ply':
                from plyfile import PlyData, PlyElement
                plydata = PlyData.read(self.voxel_path)['vertex']
                fine_points = torch.from_numpy(
                    np.stack([plydata['x'], plydata['y'], plydata['z']]).astype('float32').T)
                try:
                    self.voxel_index = torch.from_numpy(plydata['quality']).long()
                except ValueError:
                    pass
            else:
                # supporting the old version voxel points
                fine_points = torch.from_numpy(np.loadtxt(self.voxel_path)[:, 3:].astype('float32'))
        else:
            bbox = np.loadtxt(self.bbox_path)
            voxel_size = bbox[-1]
            fine_points = torch.from_numpy(bbox2voxels(bbox[:6], voxel_size))
            
        half_voxel = voxel_size * .5
        fine_length = fine_points.size(0)
 
        # transform from voxel centers to voxel corners (key/values)
        fine_coords = (fine_points / half_voxel).floor_().long()
        fine_res = (fine_points - (fine_points / half_voxel).floor_() * half_voxel).mean(0, keepdim=True)
        fine_keys0 = offset_points(fine_coords, 1.0).reshape(-1, 3)
        fine_keys, fine_feats  = torch.unique(fine_keys0, dim=0, sorted=True, return_inverse=True)
        fine_feats = fine_feats.reshape(-1, 8)
        num_keys = torch.scalar_tensor(fine_keys.size(0)).long()

        self.use_octree = getattr(args, "use_octree", False)        
        self.flatten_centers, self.flatten_children = None, None

        # assign values
        points = fine_points
        feats = fine_feats.long()
        keep = fine_feats.new_ones(fine_feats.size(0)).long()
        keys = fine_keys.long()

        # ray-marching step size
        if getattr(args, "raymarching_stepsize_ratio", 0) > 0:
            step_size = args.raymarching_stepsize_ratio * voxel_size
        else:
            step_size = args.raymarching_stepsize

        # register parameters
        self.register_buffer("points", points)   # voxel centers
        self.register_buffer("feats", feats)     # for each voxel, 8 vertexs
        self.register_buffer("keys", keys)
        self.register_buffer("keep", keep)
        self.register_buffer("num_keys", num_keys)

        self.register_buffer("voxel_size", torch.scalar_tensor(voxel_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        # set-up other hyperparameters
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        
        if shared_values is None and self.embed_dim > 0:
            self.values = Embedding(num_keys, self.embed_dim, None)
        else:
            self.values = shared_values

    def upgrade_state_dict_named(self, state_dict, name):
        # update the voxel embedding shapes
        if self.values is not None:
            loaded_values = state_dict[name + '.values.weight']
            self.values.weight = nn.Parameter(self.values.weight.new_zeros(*loaded_values.size()))
            self.values.num_embeddings = self.values.weight.size(0)
            self.total_size = self.values.weight.size(0)
            self.num_keys = self.num_keys * 0 + self.total_size
        
        if self.voxel_index is not None:
            state_dict[name + '.points'] = state_dict[name + '.points'][self.voxel_index]
            state_dict[name + '.feats'] = state_dict[name + '.feats'][self.voxel_index]
            state_dict[name + '.keep'] = state_dict[name + '.keep'][self.voxel_index]
        
        # update the buffers shapes
        self.points = self.points.new_zeros(*state_dict[name + '.points'].size())
        self.feats  = self.feats.new_zeros(*state_dict[name + '.feats'].size())
        self.keys   = self.keys.new_zeros(*state_dict[name + '.keys'].size())
        self.keep   = self.keep.new_zeros(*state_dict[name + '.keep'].size())
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--initial-boundingbox', type=str, help='the initial bounding box to initialize the model')
        parser.add_argument('--voxel-size', type=float, metavar='D', help='voxel size of the input points (initial')
        parser.add_argument('--voxel-path', type=str, help='path for pretrained voxel file. if provided no update')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--raymarching-stepsize-ratio', type=float, metavar='D',
                            help='if the concrete step size is not given (=0), we use the ratio to the voxel size as step size.')
        parser.add_argument('--use-octree', action='store_true', help='if set, instead of looping over the voxels, we build an octree.')
        
    def precompute(self, id=None, *args, **kwargs):
        feats  = self.feats[self.keep.bool()]
        points = self.points[self.keep.bool()]
        values = self.values.weight[: self.num_keys] if self.values is not None else None
        if (self.flatten_centers is None or self.flatten_children is None) and self.use_octree:
            # octree is not built. rebuild
            centers, children = build_easy_octree(points, self.voxel_size / 2.0)
            self.flatten_centers, self.flatten_children = centers, children

        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None

            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + self.num_keys * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]
        encoder_states = {
            'voxel_vertex_idx': feats,
            'voxel_center_xyz': points,
            'voxel_vertex_emb': values
        }

        if self.use_octree:
            flatten_centers, flatten_children = self.flatten_centers.clone(), self.flatten_children.clone()
            if id is not None:
                flatten_centers = flatten_centers.unsqueeze(0).expand(id.size(0), *flatten_centers.size()).contiguous()
                flatten_children = flatten_children.unsqueeze(0).expand(id.size(0), *flatten_children.size()).contiguous()
            encoder_states['voxel_octree_center_xyz'] = flatten_centers
            encoder_states['voxel_octree_children_idx'] = flatten_children
        return encoder_states

    def extract_voxels(self):
        voxel_index = torch.arange(self.keep.size(0), device=self.keep.device)
        voxel_index = voxel_index[self.keep.bool()]
        voxel_point = self.points[self.keep.bool()]
        return voxel_index, voxel_point

    def get_edge(self, ray_start, ray_dir, samples, encoder_states):
        outs = get_edge(
            ray_start + ray_dir * samples['sampled_point_depth'][:, :1], 
            encoder_states['voxel_center_xyz'].reshape(-1, 3)[samples['sampled_point_voxel_idx'][:, 0].long()], 
            self.voxel_size).type_as(ray_dir)   # get voxel edges/depth (for visualization)
        outs = (1 - outs[:, None].expand(outs.size(0), 3)) * 0.7
        return outs

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        S, V, P, _ = ray_dir.size()
        _, H, D = point_feats.size()

        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()

        if self.use_octree:  # ray-voxel intersection with SVO
            flatten_centers = encoder_states['voxel_octree_center_xyz']
            flatten_children = encoder_states['voxel_octree_children_idx']
            pts_idx, min_depth, max_depth = svo_ray_intersect(
                self.voxel_size, self.max_hits, flatten_centers, flatten_children,
                ray_start, ray_dir)
        else:   # ray-voxel intersection with all voxels
            pts_idx, min_depth, max_depth = aabb_ray_intersect(
                self.voxel_size, self.max_hits, point_xyz, ray_start, ray_dir)

        # sort the depths
        min_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth.masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        min_depth, sorted_idx = min_depth.sort(dim=-1)
        max_depth = max_depth.gather(-1, sorted_idx)
        pts_idx = pts_idx.gather(-1, sorted_idx)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + H * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    def ray_sample(self, intersection_outputs):
        min_depth = intersection_outputs['min_depth']
        max_depth = intersection_outputs['max_depth']
        pts_idx = intersection_outputs['intersected_voxel_idx']

        max_ray_length = (max_depth.masked_fill(max_depth.eq(MAX_DEPTH), 0).max(-1)[0] - min_depth.min(-1)[0]).max()
        sampled_idx, sampled_depth, sampled_dists = uniform_ray_sampling(
            pts_idx, min_depth, max_depth, self.step_size, max_ray_length, 
            self.deterministic_step or (not self.training))
        sampled_dists = sampled_dists.clamp(min=0.0)
        sampled_depth.masked_fill_(sampled_idx.eq(-1), MAX_DEPTH)
        sampled_dists.masked_fill_(sampled_idx.eq(-1), 0.0)

        samples = {
            'sampled_point_depth': sampled_depth,
            'sampled_point_distance': sampled_dists,
            'sampled_point_voxel_idx': sampled_idx,
        }
        return samples

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # encoder states
        point_feats = encoder_states['voxel_vertex_idx'] 
        point_xyz = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']

        # ray point samples
        sampled_idx = samples['sampled_point_voxel_idx'].long()
        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples['sampled_point_ray_direction']

        # prepare inputs for implicit field
        inputs = {'pos': sampled_xyz, 'ray': sampled_dir}
        if values is not None:
            # resample point features
            point_xyz = F.embedding(sampled_idx, point_xyz)
            point_feats = F.embedding(F.embedding(sampled_idx, point_feats), values).view(point_xyz.size(0), -1)

            # tri-linear interpolation
            p = ((sampled_xyz - point_xyz) / self.voxel_size + .5).unsqueeze(1)
            q = offset_points(p, .5, offset_only=True).unsqueeze(0) + .5   # BUG (FIX)
            inputs.update({'emb': trilinear_interp(p, q, point_feats)})

        return inputs

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5, encoder_states=None):
        logger.info("pruning...")
        if encoder_states is None:
            encoder_states = self.precompute(id=None)
        
        feats = encoder_states['voxel_vertex_idx'] 
        points = encoder_states['voxel_center_xyz']
        values = encoder_states['voxel_vertex_emb']
        chunk_size, bits = 64, 16

        if self.use_octree:  # clean the octree, need to be rebuilt
            self.flatten_centers, self.flatten_children = None, None

        def prune_once(feats, points, values):
            # sample points inside voxels
            sampled_xyz = offset_points(points, self.voxel_size / 2.0, bits=bits)
            sampled_idx = torch.arange(points.size(0), device=points.device)[:, None].expand(*sampled_xyz.size()[:2])
            sampled_xyz, sampled_idx = sampled_xyz.reshape(-1, 3), sampled_idx.reshape(-1)
            
            field_inputs = self.forward(
                {'sampled_point_xyz': sampled_xyz, 
                 'sampled_point_voxel_idx': sampled_idx,
                 'sampled_point_ray_direction': None}, 
                {'voxel_vertex_idx': feats,
                 'voxel_center_xyz': points,
                 'voxel_vertex_emb': values})  # get field inputs
     
            # evaluation with density
            field_outputs = field_fn(field_inputs, outputs=['sigma'])
            free_energy = -torch.relu(field_outputs['sigma']).reshape(-1, bits ** 3).max(-1)[0]
            
            # prune voxels if needed
            return (1 - torch.exp(free_energy)) > th

        keep = torch.cat([prune_once(feats[i: i + chunk_size], points[i: i + chunk_size], values) 
            for i in range(0, points.size(0), chunk_size)], 0)
        self.keep.masked_scatter_(self.keep.bool(), keep.long())
        logger.info("pruning done. # of voxels before: {}, after: {} voxels".format(points.size(0), keep.sum()))

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        encoder_states = self.precompute(id=None)
        feats, points, values = encoder_states['voxel_vertex_idx'], encoder_states['voxel_center_xyz'], encoder_states['voxel_vertex_emb']
        new_points, new_feats, new_values, new_keys = splitting_points(points, feats, values, self.voxel_size / 2.0)
        new_num_keys = new_keys.size(0)
        new_point_length = new_points.size(0)
        
        # set new voxel embeddings
        if new_values is not None:
            self.values.weight = nn.Parameter(new_values)
            self.values.num_embeddings = self.values.weight.size(0)
        
        self.total_size = new_num_keys
        self.num_keys = self.num_keys * 0 + self.total_size

        self.points = new_points
        self.feats = new_feats
        self.keep = self.keep.new_ones(new_point_length)
        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(points.size(0), self.keep.sum()))
        
    @property
    def feature_dim(self):
        return self.embed_dim

    @property
    def dummy_loss(self):
        if self.values is not None:
            return self.values.weight[0,0] * 0.0
        return 0.0
    
    @property
    def num_voxels(self):
        return self.keep.long().sum()


@register_encoder('multi_sparsevoxel_encoder')
class MultiSparseVoxelEncoder(Encoder):
    def __init__(self, args):
        super().__init__(args)

        self.voxel_lists = open(args.voxel_path).readlines()
        self.all_voxels = nn.ModuleList(
            [SparseVoxelEncoder(args, vox.strip()) for vox in self.voxel_lists])
        self.cid = None
    
    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)

    def precompute(self, id, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"
        self.cid = id[0]
        return self.all_voxels[id[0]].precompute(id, *args, **kwargs)
    
    def ray_intersect(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_intersect(*args, **kwargs)

    def ray_sample(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_sample(*args, **kwargs)

    def forward(self, samples, encoder_states):
        return self.all_voxels[self.cid].forward(samples, encoder_states)

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5):
        for id in range(len(self.all_voxels)):
           self.all_voxels[id].pruning(field_fn, th)
    
    @torch.no_grad()
    def splitting(self):
        for id in range(len(self.all_voxels)):
            self.all_voxels[id].splitting()

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim

    @property
    def dummy_loss(self):
        return sum([d.dummy_loss for d in self.all_voxels])

    @property
    def voxel_size(self):
        return self.all_voxels[0].voxel_size

    @property
    def step_size(self):
        return self.all_voxels[0].step_size

    @property
    def num_voxels(self):
        return self.all_voxels[self.cid].num_voxels


@register_encoder('shared_sparsevoxel_encoder')
class SharedSparseVoxelEncoder(Encoder):
    """
    Different from MultiSparseVoxelEncoder, we assume a shared list 
    of voxels across all models. Usually useful to learn a video sequence.  
    """
    def __init__(self, args):
        super().__init__(args)

        # using a shared voxel
        self.voxel_path = args.voxel_path
        self.num_frames = args.num_frames
        self.all_voxels = [SparseVoxelEncoder(args, self.voxel_path)]
        self.all_voxels =  nn.ModuleList(self.all_voxels + [
            SparseVoxelEncoder(args, self.voxel_path, shared_values=self.all_voxels[0].values)
            for i in range(self.num_frames - 1)])
        self.context_embed_dim = args.context_embed_dim
        self.contexts = nn.Embedding(self.num_frames, self.context_embed_dim, None)
        self.cid = None
    
    def precompute(self, id, *args, **kwargs):
        # TODO: this is a HACK for simplicity
        assert id.size(0) == 1, "for now, only works for one object"
        self.cid = id[0]
        return self.all_voxels[id[0]].precompute(id, *args, **kwargs)

    def ray_intersect(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_intersect(*args, **kwargs)

    def ray_sample(self, *args, **kwargs):
        return self.all_voxels[self.cid].ray_sample(*args, **kwargs)

    def forward(self, samples, encoder_states):
        inputs = self.all_voxels[self.cid].forward(samples, encoder_states)
        inputs.update({'context': self.contexts(self.cid).unsqueeze(0)})
        return inputs

    @torch.no_grad()
    def pruning(self, field_fn, th=0.5):
        for cid in range(len(self.all_voxels)):
           id = torch.tensor([cid], device=self.contexts.weight.device)
           self.all_voxels[cid].pruning(field_fn, th, 
                encoder_states={name: v[0] for name, v in self.precompute(id).items()})

    @torch.no_grad()
    def splitting(self):
        logger.info("splitting...")
        all_feats, all_points = [], []
        for id in range(len(self.all_voxels)):
            feats, points, values = self.all_voxels[id].precompute(id=None)
            all_feats.append(feats)
            all_points.append(points)
        feats, points = torch.cat(all_feats, 0), torch.cat(all_points, 0)
        unique_feats, unique_idx = torch.unique(feats, dim=0, return_inverse=True)
        unique_points = points[
            unique_feats.new_zeros(unique_feats.size(0)).scatter_(
                0, unique_idx, torch.arange(unique_idx.size(0), device=unique_feats.device)
        )]
        new_points, new_feats, new_values = splitting_points(unique_points, unique_feats, values, self.voxel_size / 2.0)
        new_num_keys = new_values.size(0)
        new_point_length = new_points.size(0)

        # set new voxel embeddings (shared voxels)
        self.all_voxels[0].values.weight = nn.Parameter(new_values)
        self.all_voxels[0].values.num_embeddings = new_num_keys

        for id in range(len(self.all_voxels)):
            self.all_voxels[id].total_size = new_num_keys
            self.all_voxels[id].num_keys = self.all_voxels[id].num_keys * 0 + self.all_voxels[id].total_size

            self.all_voxels[id].points = new_points
            self.all_voxels[id].feats = new_feats
            self.all_voxels[id].keep = self.all_voxels[id].keep.new_ones(new_point_length)

        logger.info("splitting done. # of voxels before: {}, after: {} voxels".format(
            unique_points.size(0), new_point_length))

    @property
    def feature_dim(self):
        return self.all_voxels[0].embed_dim + self.context_embed_dim

    @property
    def dummy_loss(self):
        return sum([d.dummy_loss for d in self.all_voxels])

    @property
    def voxel_size(self):
        return self.all_voxels[0].voxel_size

    @property
    def step_size(self):
        return self.all_voxels[0].step_size

    @property
    def num_voxels(self):
        return self.all_voxels[self.cid].num_voxels

    @staticmethod
    def add_args(parser):
        SparseVoxelEncoder.add_args(parser)
        parser.add_argument('--num-frames', type=int, help='the total number of frames')
        parser.add_argument('--context-embed-dim', type=int, help='context embedding for each view')


@register_encoder('triangle_mesh_encoder')
class TriangleMeshEncoder(SparseVoxelEncoder):
    """
    Training on fixed mesh model. Cannot pruning..
    """
    def __init__(self, args, mesh_path=None, shared_values=None):
        super(SparseVoxelEncoder, self).__init__(args)
        self.mesh_path = mesh_path if mesh_path is not None else args.mesh_path
        assert (self.mesh_path is not None) and os.path.exists(self.mesh_path)
        
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(self.mesh_path)
        vertices = torch.from_numpy(np.asarray(mesh.vertices, dtype=np.float32))
        faces = torch.from_numpy(np.asarray(mesh.triangles, dtype=np.long))
    
        step_size = args.raymarching_stepsize
        cage_size = step_size * 10  # truncated space around the triangle surfaces
        self.register_buffer("cage_size", torch.scalar_tensor(cage_size))
        self.register_buffer("step_size", torch.scalar_tensor(step_size))
        self.register_buffer("max_hits", torch.scalar_tensor(args.max_hits))

        self.vertices = nn.Parameter(vertices, requires_grad=getattr(args, "trainable_vertices", False))
        self.faces = nn.Parameter(faces, requires_grad=False)

        # set-up other hyperparameters
        self.embed_dim = getattr(args, "voxel_embed_dim", None)
        self.deterministic_step = getattr(args, "deterministic_step", False)
        self.values = None
        self.blur_ratio = getattr(args, "blur_ratio", 0.0)

    def upgrade_state_dict_named(self, state_dict, name):
        pass
    
    @staticmethod
    def add_args(parser):
        parser.add_argument('--mesh-path', type=str, help='path for initial mesh file')
        parser.add_argument('--voxel-embed-dim', type=int, metavar='N', help="embedding size")
        parser.add_argument('--deterministic-step', action='store_true',
                            help='if set, the model runs fixed stepsize, instead of sampling one')
        parser.add_argument('--max-hits', type=int, metavar='N', help='due to restrictions we set a maximum number of hits')
        parser.add_argument('--raymarching-stepsize', type=float, metavar='D', 
                            help='ray marching step size for sparse voxels')
        parser.add_argument('--blur-ratio', type=float, default=0,
                            help="it is possible to shoot outside the triangle. default=0")
        parser.add_argument('--trainable-vertices', action='store_true',
                            help='if set, making the triangle trainable. experimental code. not ideal.')

    def precompute(self, id=None, *args, **kwargs):
        feats, points, values = self.faces, self.vertices, self.values
        if id is not None:
            # extend size to support multi-objects
            feats  = feats.unsqueeze(0).expand(id.size(0), *feats.size()).contiguous()
            points = points.unsqueeze(0).expand(id.size(0), *points.size()).contiguous()
            values = values.unsqueeze(0).expand(id.size(0), *values.size()).contiguous() if values is not None else None
            
            # moving to multiple objects
            if id.size(0) > 1:
                feats = feats + points.size(1) * torch.arange(id.size(0), 
                    device=feats.device, dtype=feats.dtype)[:, None, None]

        encoder_states = {
            'mesh_face_vertex_idx': feats,
            'mesh_vertex_xyz': points,
        }
        return encoder_states

    def get_edge(self, ray_start, ray_dir, *args, **kwargs):
        return torch.ones_like(ray_dir) * 0.7

    @property
    def voxel_size(self):
        return self.cage_size

    def ray_intersect(self, ray_start, ray_dir, encoder_states):
        point_xyz = encoder_states['mesh_vertex_xyz']
        point_feats =encoder_states['mesh_face_vertex_idx']
        
        S, V, P, _ = ray_dir.size()
        F, G = point_feats.size(1), point_xyz.size(1)
  
        # ray-voxel intersection
        ray_start = ray_start.expand_as(ray_dir).contiguous().view(S, V * P, 3).contiguous()
        ray_dir = ray_dir.reshape(S, V * P, 3).contiguous()
        pts_idx, depth, uv = triangle_ray_intersect(
            self.cage_size, self.blur_ratio, self.max_hits, point_xyz, point_feats, ray_start, ray_dir)
        min_depth = (depth[:,:,:,0] + depth[:,:,:,1]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        max_depth = (depth[:,:,:,0] + depth[:,:,:,2]).masked_fill_(pts_idx.eq(-1), MAX_DEPTH)
        hits = pts_idx.ne(-1).any(-1)  # remove all points that completely miss the object
        
        if S > 1:  # extend the point-index to multiple shapes (just in case)
            pts_idx = (pts_idx + G * torch.arange(S, 
                device=pts_idx.device, dtype=pts_idx.dtype)[:, None, None]
                ).masked_fill_(pts_idx.eq(-1), -1)

        intersection_outputs = {
            "min_depth": min_depth,
            "max_depth": max_depth,
            "intersected_voxel_idx": pts_idx
        }
        return ray_start, ray_dir, intersection_outputs, hits

    @torch.enable_grad()
    def forward(self, samples, encoder_states):
        # TODO: enable mesh embedding learning

        sampled_xyz = samples['sampled_point_xyz'].requires_grad_(True)
        sampled_dir = samples['sampled_point_ray_direction']

        # prepare inputs for implicit field
        inputs = {'pos': sampled_xyz, 'ray': sampled_dir}
        return inputs

    @property
    def num_voxels(self):
        return self.vertices.size(0)

def bbox2voxels(bbox, voxel_size):
    vox_min, vox_max = bbox[:3], bbox[3:]
    steps = ((vox_max - vox_min) / voxel_size).round().astype('int64') + 1
    x, y, z = [c.reshape(-1).astype('float32') for c in np.meshgrid(np.arange(steps[0]), np.arange(steps[1]), np.arange(steps[2]))]
    x, y, z = x * voxel_size + vox_min[0], y * voxel_size + vox_min[1], z * voxel_size + vox_min[2]
    return np.stack([x, y, z]).T.astype('float32')


