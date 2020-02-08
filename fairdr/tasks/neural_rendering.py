# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import FairseqTask, register_task
from fairdr.data import RenderedImageDataset, SampledPixelDataset, WorldCoordDataset


@register_task("single_object_rendering")
class SingleObjRenderingTask(FairseqTask):
    """
    Task for remembering & rendering a single object.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser"""
        parser.add_argument("data", help='data-path or data-directoy')
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
        self.datasets[split] = RenderedImageDataset(
            self.args.data,
            self.args.max_train_view,
            self.args.view_per_batch,
            self.args.view_resolution,
            train=(split == 'train'))
        
        if split != 'test':
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