# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq.tasks import FairseqTask, register_task
from fairdr.data import RenderedImageDataset


@register_task("single_object_rendering")
class SingleObjRenderingTask(FairseqTask):
    """
    Task for remembering & rendering a single object.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser"""
        parser.add_argument("data", help='data-path or data-directoy')
        parser.add_argument("--train-view", type=int, default=50, 
                            help='number of views sampled for each shape')

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
            self.args.train_view,
            train=(split == 'train'))

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None
        