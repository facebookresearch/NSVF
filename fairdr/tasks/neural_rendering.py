# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import torch
import numpy as np
from collections import defaultdict

from argparse import Namespace

from fairseq.tasks import FairseqTask, register_task
from fairseq.optim.fp16_optimizer import FP16Optimizer

from fairdr.data import (
    ShapeViewDataset, SampledPixelDataset, 
    WorldCoordDataset, ShapeDataset, InfiniteDataset
)
from fairdr.data.data_utils import write_images
from fairdr.renderer import NeuralRenderer
from fairdr.data.trajectory import get_trajectory


@register_task("single_object_rendering")
class SingleObjRenderingTask(FairseqTask):
    """
    Task for remembering & rendering a single object.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser"""
        parser.add_argument("data", help='data-path or data-directoy')
        parser.add_argument("--no-preload", action="store_true")
        parser.add_argument("--no-load-binary", action="store_true")
        parser.add_argument("--load-depth", action="store_true", 
                            help="load depth images if exists")
        parser.add_argument("--transparent-background", type=float, default=-0.8,
                            help="background color if the image is transparent")
        parser.add_argument("--load-point", action="store_true",
                            help="load sparse voxel generated by visual-hull if exists")
        parser.add_argument("--load-mask", action="store_true",
                            help="load pre-computed masks which is useful for subsampling during training.")
        parser.add_argument("--max-train-view", type=int, default=50, 
                            help="number of views sampled for training, can be unlimited if set -1")
        parser.add_argument("--max-valid-view", type=int, default=50,
                            help="number of views sampled for validation, can be unlimited if set -1")
        parser.add_argument("--subsample-valid", type=int, default=-1,
                            help="if set > -1, subsample the validation (when training set is too large)")
        parser.add_argument("--view-per-batch", type=int, default=6,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--valid-view-per-batch", type=int, default=6,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--pixel-per-view", default=None, nargs='?', const=16384, type=int,
                            help="how many pixels to sample from each view. -1 means using all pixels")
        parser.add_argument("--sampling-on-mask", default=1.0, nargs='?', const=0.9, type=float,
                            help="this value determined the probability of sampling rays on masks")
        parser.add_argument("--sampling-on-bbox", action='store_true',
                            help="sampling points to close to the mask")
        parser.add_argument("--sampling-patch-size", type=int, default=1, 
                            help="sample pixels based on patches instead of independent pixels")
        parser.add_argument("--view-resolution", type=int, default=64,
                            help="width for the squared image. downsampled from the original.")       
        parser.add_argument("--min-color", choices=(0, -1), default=-1, type=int,
                            help="RGB range used in the model. conventionally used -1 ~ 1")
        parser.add_argument("--virtual-epoch-steps", type=int, default=None,
                            help="virtual epoch used in Infinite Dataset. if None, set max-update")
        parser.add_argument("--pruning-every-steps", type=int, default=None,
                            help="if the model supports pruning, prune unecessary voxels")
        parser.add_argument("--half-voxel-size-at", type=str, default=None,
                            help='specific detailed number of updates to half the voxel sizes')
        parser.add_argument("--reduce-step-size-at", type=str, default=None,
                            help='specific detailed number of updates to reduce the raymarching step sizes')
        parser.add_argument("--rendering-every-steps", type=int, default=None,
                            help="if set, enables rendering online with default parameters")
        parser.add_argument("--rendering-args", type=str, metavar='JSON')
        parser.add_argument("--pruning-th", type=float, default=0.5,
                            help="if larger than this, we choose keep the voxel.")

    def __init__(self, args):
        super().__init__(args)
        
        if len(self.args.tensorboard_logdir) > 0 and getattr(args, "distributed_rank", -1) == 0:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.args.tensorboard_logdir + '/images')
        else:
            self.writer = None

        self._num_updates = 0
        self._safe_steps = 0

        self.pruning_every_steps = getattr(self.args, "pruning_every_steps", None)
        self.pruning_th = getattr(self.args, "pruning_th", 0.5)
        self.rendering_every_steps = getattr(self.args, "rendering_every_steps", None)
        self.steps_to_half_voxels = getattr(self.args, "half_voxel_size_at", None)
        self.steps_to_reduce_step = getattr(self.args, "reduce_step_size_at", None)
        
        if self.steps_to_half_voxels is not None:
            self.steps_to_half_voxels = [int(s) for s in self.steps_to_half_voxels.split(',')]
        if self.steps_to_reduce_step is not None:
            self.steps_to_reduce_step = [int(s) for s in self.steps_to_reduce_step.split(',')]
           
        if self.rendering_every_steps is not None:
            gen_args = {
                'path': args.save_dir,
                'render_beam': 1, 'render_resolution': 512,
                'render_num_frames': 120, 'render_angular_speed': 3,
                'render_output_types': ["rgb"], 'render_raymarching_steps': 10,
                'render_at_vector': "(0,0,0)", 'render_up_vector': "(0,0,-1)",
                'render_path_args': "{'radius': 1.5, 'h': 0.5}",
                'render_path_style': 'circle', "render_output": None
            }
            gen_args.update(json.loads(getattr(args, 'rendering_args', '{}') or '{}'))
            self.renderer = self.build_generator(Namespace(**gen_args))    
        else:
            self.renderer = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        Setup the task
        """
        return cls(args)

    def repeat_dataset(self, split):
        max_view = self.args.max_train_view if split == 'train' \
                else self.args.max_valid_view
        return int(np.ceil(max_view / self.args.view_per_batch / 
            self.args.distributed_world_size) * self.args.distributed_world_size)

    def load_dataset(self, split, **kwargs):
        """
        Load a given dataset split (train, valid, test)
        """
        
        if split != 'test':
            self.datasets[split] = ShapeViewDataset(
                self.args.data,
                max_train_view=self.args.max_train_view,
                max_valid_view=self.args.max_valid_view,
                subsample_valid=self.args.subsample_valid 
                    if split == 'valid' else -1,
                num_view=self.args.view_per_batch 
                    if split == 'train' 
                    else self.args.valid_view_per_batch,
                resolution=self.args.view_resolution,
                train=(split == 'train'),
                load_depth=self.args.load_depth,
                load_mask=self.args.load_mask,
                load_point=self.args.load_point,
                repeat=self.repeat_dataset(split),
                preload=(not getattr(self.args, "no_preload", False)),
                binarize=(not getattr(self.args, "no_load_binary", False)),
                bg_color=getattr(self.args, "transparent_background", -0.8),
                min_color=getattr(self.args, "min_color", -1))

            if split == 'train' and (self.args.pixel_per_view is not None):
                self.datasets[split] = SampledPixelDataset(
                    self.datasets[split],
                    self.args.pixel_per_view,
                    self.args.sampling_on_mask,
                    self.args.sampling_on_bbox,
                    self.args.view_resolution,
                    self.args.sampling_patch_size)
            self.datasets[split] = WorldCoordDataset(
                self.datasets[split]
            )

            if split == 'train':   # infinite sampler
                max_step = getattr(self.args, "virtual_epoch_steps", None)
                if max_step is not None:
                    total_num_models = max_step * self.args.distributed_world_size * self.args.max_sentences
                    self.datasets[split] = InfiniteDataset(
                        self.datasets[split], total_num_models)

        else:
            self.datasets[split] = ShapeDataset(
                self.args.data,
                load_point=self.args.load_point)

    def build_generator(self, args):
        """
        build a neural renderer for visualization
        """
        return NeuralRenderer(
            beam=args.render_beam,
            resolution=args.render_resolution,
            frames=args.render_num_frames,
            speed=args.render_angular_speed,
            raymarching_steps=args.render_raymarching_steps,
            path_gen=get_trajectory(args.render_path_style)(
                **eval(args.render_path_args)
            ),
            at=eval(args.render_at_vector),
            up=eval(args.render_up_vector),
            output_dir=args.render_output if args.render_output is not None
                else os.path.join(args.path, "output"),
            output_type=args.render_output_types
        )

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None
    
    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None

    def update_step(self, num_updates):
        """Task level update when number of updates increases.

        This is called after the optimization step and learning rate
        update at each iteration.
        """
        self._num_updates = num_updates

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self._num_updates = update_num
    
        if self.steps_to_half_voxels is not None and \
            self._num_updates in self.steps_to_half_voxels:
            
            model.adjust('split')
            if isinstance(optimizer, FP16Optimizer):
                optimizer.fp32_params.data.copy_(FP16Optimizer.build_fp32_params(optimizer.fp16_params, True).data)  
    
            # reset optimizer, forget the optimizer history after pruning.
            # avoid harmful history of adam
            for p in optimizer.optimizer.state:
                for key in optimizer.optimizer.state[p]:
                    if key != 'step':
                        optimizer.optimizer.state[p][key] *= 0.0
            
            # set safe steps. do not update parameters, accumulate Adam state
            self._safe_steps = 0

        if self.pruning_every_steps is not None and \
            (self._num_updates % self.pruning_every_steps == 0) and \
            (self._num_updates > 0):
            model.eval()
            model.adjust('prune', id=sample['id'], th=self.pruning_th)

        if self.rendering_every_steps is not None and \
            (self._num_updates % self.rendering_every_steps == 0) and \
            (self._num_updates > 0) and \
            self.renderer is not None:

            outputs = self.inference_step(self.renderer, [model], [sample, 0])[1]
            if getattr(self.args, "distributed_rank", -1) == 0:  # save only for master
                self.renderer.save_images(outputs, self._num_updates)
        
        if self.steps_to_reduce_step is not None and \
            self._num_updates in self.steps_to_reduce_step:
            model.adjust('reduce')

        if self._safe_steps > 0:  # do not update parameters, accumulate Adam state
            optimizer.set_lr(0.0)
            self._safe_steps -= 1

        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        # model.pruning(points=sample['points'])
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.writer is not None:
            images = model.visualize(
                sample,
                shape=0, view=0,
                target_map=True,
                depth_map=True, 
                normal_map=True, 
                error_map=False,
                hit_map=True)

            if images is not None:
                write_images(self.writer, images, self._num_updates)
        
        return loss, sample_size, logging_output
    

@register_task("sequence_object_rendering")
class SequenceObjRenderingTask(SingleObjRenderingTask):
    """
    Task for rendering a sequence of an single object.
       -- it is a conditional generation.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments"""    
        SingleObjRenderingTask.add_args(parser)

    def repeat_dataset(self, split):
        return 1

    def load_dataset(self, split, **kwargs):
        super().load_dataset(split, **kwargs)

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        self._num_updates = update_num

        if self.rendering_every_steps is not None and \
            (self._num_updates % self.rendering_every_steps == 0) and \
            (self._num_updates > 0) and \
            self.renderer is not None:

            outputs = self.inference_step(self.renderer, [model], [sample, 0])[1]
            if getattr(self.args, "distributed_rank", -1) == 0:  # save only for master
                self.renderer.save_images(outputs, self._num_updates)

        if self.steps_to_half_voxels is not None and \
                self._num_updates in self.steps_to_half_voxels:
            model.adjust('level')

        return super(SingleObjRenderingTask, self).train_step(sample, model, criterion, optimizer, update_num, ignore_grad)