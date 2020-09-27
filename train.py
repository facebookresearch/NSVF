#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from fairnr_cli.train import cli_main
from fairnr_cli.launch_slurm import launch

if __name__ == '__main__':
    all_args = sys.argv[1:]
    try:
        # simplified code for launching on slurm-based clusters
        # it can naturally adapt to local training with almost the same scripts
        # For more complicated functions (for example, sweep parameters), 
        # Please use fairseq sweep.
        
        slurm_idx = all_args.index('--slurm-args')
        slurm_arg = eval(all_args[slurm_idx+1])
        all_args = all_args[:slurm_idx] + all_args[slurm_idx+2:]
        launch(slurm_arg, all_args)
    
    except ValueError:
        cli_main()
