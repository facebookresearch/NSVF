# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import torch
import numpy as np

from fairseq.tasks import FairseqTask, register_task
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
        parser.add_argument("--load-depth", action="store_true", 
                            help="load depth images if exists")
        parser.add_argument("--load-point", action="store_true",
                            help="load sparse voxel generated by visual-hull if exists")
        parser.add_argument("--load-mask", action="store_true",
                            help="load pre-computed masks which is useful for subsampling during training.")
        parser.add_argument("--max-train-view", type=int, default=50, 
                            help="number of views sampled for training, can be unlimited if set -1")
        parser.add_argument("--max-valid-view", type=int, default=50,
                            help="number of views sampled for validation, can be unlimited if set -1")
        parser.add_argument("--view-per-batch", type=int, default=6,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--pixel-per-view", default=None, nargs='?', const=16384, type=int,
                            help="how many pixels to sample from each view. -1 means using all pixels")
        parser.add_argument("--sampling-on-mask", default=1.0, nargs='?', const=0.9, type=float,
                            help="this value determined the probability of sampling rays on masks")
        parser.add_argument("--sampling-on-bbox", action='store_true')
        parser.add_argument("--view-resolution", type=int, default=64,
                            help="height/width for the squared image. downsampled from the original.")

    def __init__(self, args):
        super().__init__(args)
        
        if len(self.args.tensorboard_logdir) > 0 and getattr(args, "distributed_rank", -1) == 0:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.args.tensorboard_logdir + '/images')
        else:
            self.writer = None

        self._num_updates = 0

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
                num_view=self.args.view_per_batch,
                resolution=self.args.view_resolution,
                train=(split == 'train'),
                load_depth=self.args.load_depth,
                load_mask=self.args.load_mask,
                load_point=self.args.load_point,
                repeat=self.repeat_dataset(split),
                preload=(not getattr(self.args, "no_preload", False)))

            if split == 'train' and (self.args.pixel_per_view is not None):
                self.datasets[split] = SampledPixelDataset(
                    self.datasets[split],
                    self.args.pixel_per_view,
                    self.args.sampling_on_mask,
                    self.args.sampling_on_bbox)
            self.datasets[split] = WorldCoordDataset(
                self.datasets[split]
            )

            if split == 'train':   # infinite sampler
                total_num_models = self.args.max_update * self.args.distributed_world_size * self.args.max_sentences
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
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
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
        assert os.path.isfile(self.args.data), \
            "a list of multiple objects must be saved in a document"
        assert self.args.load_point, "for now only supports point condition"  # TODO

        super().load_dataset(split, **kwargs)
