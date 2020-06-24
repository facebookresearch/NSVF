# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import types
from itertools import chain

import torch
import torch.optim
import torch.distributed as dist

from fairseq.optim import FairseqOptimizer, register_optimizer
from fairseq.optim.adam import FairseqAdam

logger = logging.getLogger(__name__)


@register_optimizer('fairnr_adam')
class FairNRAdam(FairseqOptimizer):
    """
    This is a Wrapper for standard FairseqAdam which supports reset.
    For now, it only supports FP32
    """
    def __init__(self, args, params):
        super().__init__(args)
        self.adam = FairseqAdam(args, params)
        self._optimizer = self.adam._optimizer

    @staticmethod
    def add_args(parser):
        FairseqAdam.add_args(parser)

    @property
    def optimizer_config(self):
        return self.adam.optimizer_config

    def reset_optimizer(self, model, criterion):
        new_params = list(
                filter(
                    lambda p: p.requires_grad,
                    chain(model.parameters(), criterion.parameters()),
                )
            )
        # reset and recreate a new optimizer
        self.adam = FairseqAdam(self.args, new_params)
        self._optimizer = self.adam._optimizer
        