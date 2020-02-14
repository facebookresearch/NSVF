# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import numpy as np

from fairseq.tasks import FairseqTask, register_task
from fairdr.data import RenderedImageDataset, SampledPixelDataset, WorldCoordDataset
from fairdr.data.data_utils import write_images


@register_task("single_object_rendering")
class SingleObjRenderingTask(FairseqTask):
    """
    Task for remembering & rendering a single object.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser"""
        parser.add_argument("data", help='data-path or data-directoy')
        parser.add_argument("--load-depth", action="store_true", help="load depth images if exists")
        parser.add_argument("--max-train-view", type=int, default=50, 
                            help="number of views sampled for training, can be unlimited if set -1")
        parser.add_argument("--view-per-batch", type=int, default=6,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--pixel-per-view", type=int, default=1024,
                            help="how many pixels to sample from each view")
        parser.add_argument("--view-resolution", type=int, default=64,
                            help="height/width for the squared image. downsampled from the original.")

    def __init__(self, args):
        super().__init__(args)

        if args.distributed_rank == 0:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.args.tensorboard_logdir)
        else:
            self.writer = None

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        Setup the task
        """
        return cls(args)

    def load_dataset(self, split, **kwargs):
        """
        Load a given dataset split (train, valid, test)
        """
        repeats = int(np.ceil(self.args.max_train_view / 
                        self.args.view_per_batch / 
                        self.args.distributed_world_size
                        ) * self.args.distributed_world_size)

        self.datasets[split] = RenderedImageDataset(
            self.args.data,
            self.args.max_train_view,
            self.args.view_per_batch,
            self.args.view_resolution,
            train=(split == 'train'),
            load_depth=self.args.load_depth,
            repeat=repeats)
        
        if split == 'train' and (self.args.pixel_per_view > 0):
            self.datasets[split] = SampledPixelDataset(
                self.datasets[split],
                self.args.pixel_per_view)

        self.datasets[split] = WorldCoordDataset(
            self.datasets[split]
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

    def valid_step(self, sample, model, criterion):
        # from fairseq.pdb import set_trace
        # set_trace()
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        
        # visualize in tensorboard
        for i in [0, 1, 2]:
            if i >= sample['alpha'].size(1):
                continue
            images = model.visualize(sample, 0, i)
            if images is not None and self.args.distributed_rank == 0:
                write_images(self.writer, images)

        return loss, sample_size, logging_output
    