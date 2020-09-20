#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
This is a copy of fairseq-generate while simpler for other usage.
"""


import logging
import math
import os
import sys
import time
import torch
import imageio
import numpy as np

from fairseq import checkpoint_utils, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
from fairnr import options


def main(args):
    assert args.path is not None, '--path required for generation!'

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(args.results_path, 'generate-{}.txt'.format(args.gen_subset))
        with open(output_path, 'w', buffering=1) as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def _main(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fairnr_cli.render')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)


    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)
    

    output_files, step= [], 0
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for i, sample in enumerate(t):        
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            gen_timer.start()
           
            step, _output_files = task.inference_step(generator, models, [sample, step])
            output_files += _output_files
        
            gen_timer.stop(500)
            wps_meter.update(500)
            t.log({'wps': round(wps_meter.avg)})
            
            break
            # if i > 5:
            #     break

    generator.save_images(output_files, combine_output=args.render_combine_output)

def cli_main():
    parser = options.get_rendering_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
