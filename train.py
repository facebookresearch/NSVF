#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys, os
from fairnr_cli.train import cli_main
from fairnr_cli.launch_slurm import launch

if __name__ == '__main__':
    if os.getenv('SLURM_ARGS') is not None:
        slurm_arg = eval(os.getenv('SLURM_ARGS'))
        all_args = sys.argv[1:]

        print(slurm_arg)
        print(all_args)
        launch(slurm_arg, all_args)
    
    else:
        cli_main()
