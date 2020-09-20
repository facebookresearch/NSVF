# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Base classes for various  models.

The basic principle of differentiable rendering is two components:
    -- an field or so-called geometric field (GE)
    -- an raymarcher or so-called differentiable ray-marcher (RM)
So it can be composed as a GERM model
"""

import logging
import torch
import torch.nn as nn
import skimage.metrics
import imageio, os
import numpy as np

from fairseq.models import BaseFairseqModel
from fairnr.modules.encoder import get_encoder
from fairnr.modules.field import get_field
from fairnr.modules.renderer import get_renderer
from fairnr.modules.reader import get_reader
from fairnr.data.geometry import ray, compute_normal_map, compute_normal_map
from fairnr.data.data_utils import recover_image

logger = logging.getLogger(__name__)


class BaseModel(BaseFairseqModel):
    """Base class"""

    ENCODER = 'abstract_encoder'
    FIELD = 'abstract_field'
    RAYMARCHER = 'abstract_renderer'
    READER = 'abstract_reader'

    def __init__(self, args, reader, encoder, field, raymarcher):
        super().__init__()
        self.args = args
        self.reader = reader
        self.encoder = encoder
        self.field = field
        self.raymarcher = raymarcher
        self.cache = None

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        reader = get_reader(cls.READER)(args)
        encoder = get_encoder(cls.ENCODER)(args)
        field = get_field(cls.FIELD)(args)
        raymarcher = get_renderer(cls.RAYMARCHER)(args)
        return cls(args, reader, encoder, field, raymarcher)

    @classmethod
    def add_args(cls, parser):
        get_reader(cls.READER).add_args(parser)
        get_renderer(cls.RAYMARCHER).add_args(parser)
        get_encoder(cls.ENCODER).add_args(parser)
        get_field(cls.FIELD).add_args(parser)

    @property
    def dummy_loss(self):
        return sum([p.sum() for p in self.parameters()]) * 0.0

    # def forward(self, ray_start, ray_dir, ray_split=1, **kwargs):
    def forward(self, ray_split=1, **kwargs):
        ray_start, ray_dir, uv = self.reader(**kwargs)

        if ray_split == 1:
            results = self._forward(ray_start, ray_dir, **kwargs)
        else:
            total_rays = ray_dir.shape[2]
            chunk_size = total_rays // ray_split
            results = [
                self._forward(
                    ray_start, ray_dir[:, :, i: i+chunk_size], **kwargs)
                for i in range(0, total_rays, chunk_size)
            ]
            results = self.merge_outputs(results)

        if results.get('sampled_uv', None) is None:
            results['sampled_uv'] = uv
        results['ray_start'] = ray_start
        results['ray_dir'] = ray_dir

        # caching the prediction
        self.cache = {
            w: results[w].detach() 
                if isinstance(w, torch.Tensor) 
                else results[w] 
            for w in results
        }
        return results

    def _forward(self, ray_start, ray_dir, **kwargs):
        raise NotImplementedError
    
    def merge_outputs(self, outputs):
        new_output = {}
        for key in outputs[0]:
            if isinstance(outputs[0][key], torch.Tensor) and outputs[0][key].dim() > 2:
                new_output[key] = torch.cat([o[key] for o in outputs], 2)
            else:
                new_output[key] = outputs[0][key]        
        return new_output

    @torch.no_grad()
    def visualize(self, sample, output=None, shape=0, view=0, **kwargs):
        width = int(sample['size'][shape, view][1].item())
        img_id = '{}_{}'.format(sample['shape'][shape], sample['view'][shape, view])
        
        if output is None:
            assert self.cache is not None, "need to run forward-pass"
            output = self.cache  # make sure to run forward-pass.

        images = {}
        images = self._visualize(images, sample, output, [img_id, shape, view, width, 'render'])
        images = self._visualize(images, sample, sample, [img_id, shape, view, width, 'target'])
        images = {
            tag: recover_image(width=width, **images[tag]) 
                for tag in images if images[tag] is not None
        }
        return images
        
    def _visualize(self, images, sample, output, state, **kwargs):
        img_id, shape, view, width, name = state
        if 'colors' in output and output['colors'] is not None:
            images['{}_color/{}:HWC'.format(name, img_id)] ={
                'img': output['colors'][shape, view]}
        
        if 'depths' in output and output['depths'] is not None:
            min_depth, max_depth = output['depths'].min(), output['depths'].max()
            images['{}_depth/{}:HWC'.format(name, img_id)] = {
                'img': output['depths'][shape, view], 
                'min_val': min_depth, 
                'max_val': max_depth}

            normals = compute_normal_map(
                output['ray_start'][shape, view].float(),
                output['ray_dir'][shape, view].float(),
                output['depths'][shape, view].float(),
                sample['extrinsics'][shape, view].float().inverse(), width)
            images['{}_normal/{}:HWC'.format(name, img_id)] = {
                'img': normals, 'min_val': -1, 'max_val': 1}
        
        return images

    def add_eval_scores(self, logging_output, sample, output, criterion, scores=['ssim', 'psnr', 'lpips'], outdir=None):
        predicts, targets = output['colors'], sample['colors']
        ssims, psnrs, lpips, rmses = [], [], [], []
        
        for s in range(predicts.size(0)):
            for v in range(predicts.size(1)):
                width = int(sample['size'][s, v][1])
                p = recover_image(predicts[s, v], width=width)
                t = recover_image(targets[s, v], width=width)
                pn, tn = p.numpy(), t.numpy()
                if 'ssim' in scores:
                    ssims += [skimage.metrics.structural_similarity(pn, tn, multichannel=True, data_range=1)]
                if 'psnr' in scores:
                    psnrs += [skimage.metrics.peak_signal_noise_ratio(pn, tn, data_range=1)]
                if 'lpips' in scores and hasattr(criterion, 'lpips'):
                    with torch.no_grad():
                        lpips += [criterion.lpips.forward(
                            p.unsqueeze(-1).permute(3,2,0,1),
                            t.unsqueeze(-1).permute(3,2,0,1),
                            normalize=True).item()]
                if 'depths' in sample:
                    td = sample['depths'][sample['depths'] > 0]
                    pd = output['depths'][sample['depths'] > 0]
                    rmses += [torch.sqrt(((td - pd) ** 2).mean()).item()]

                if outdir is not None:
                    def imsave(filename, image):
                        imageio.imsave(os.path.join(outdir, filename), (image * 255).astype('uint8'))
                    
                    figname = '-{:03d}_{:03d}.png'.format(sample['id'][s], sample['view'][s, v])
                    imsave('output' + figname, pn)
                    imsave('target' + figname, tn)
                    imsave('normal' + figname, recover_image(compute_normal_map(
                        output['ray_start'][s, v].float(), output['ray_dir'][s, v].float(),
                        output['depths'][s, v].float(), sample['extrinsics'][s, v].float().inverse(), width=width),
                        min_val=-1, max_val=1, width=width).numpy())
                    
        if len(ssims) > 0:
            logging_output['ssim_loss'] = np.mean(ssims)
        if len(psnrs) > 0:
            logging_output['psnr_loss'] = np.mean(psnrs)
        if len(lpips) > 0:
            logging_output['lpips_loss'] = np.mean(lpips)
        if len(rmses) > 0:
            logging_output['rmses_loss'] = np.mean(rmses)

    def adjust(self, **kwargs):
        raise NotImplementedError

    @property
    def text(self):
        return "fairnr BaseModel"


