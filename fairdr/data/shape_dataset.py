# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
from fairseq.data import FairseqDataset

# from . import data_utils
# from .collaters import Seq2SeqCollater


class RenderedImageDataset(FairseqDataset):
    """
    A dataset contains a series of images renderred offline for an object.
    """

    def __init__(self, paths, train_view, train=True):
        if os.path.isdir(paths):
            self.paths = [paths]
        else:
            self.paths = [line.strip() for line in open(paths)]
        
        self.train = train
        self.train_view = train_view

    def ordered_indices(self):
        return np.arange(len(self.paths))

    def __getitem__(self, index):
        


        from fairseq.pdb import set_trace
        set_trace()