# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F


class UniformSearchRayMarcher(nn.Module):

    """
    Uniform-search requires a value representing occupacy
    It is costly, but esay to parallize if we have GPUs.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args

    def search(occupacy_handle, start, ray_dir, end, steps, tau=0.5): 
        from faiseq.pdb import set_trace;
        set_trace()
        pass