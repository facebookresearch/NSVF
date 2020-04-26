#!/usr/bin/env python3 -u
#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import numpy as np
import torch
from itertools import chain
from fairseq import checkpoint_utils, distributed_utils, options, utils
from fairseq.logging import metrics, progress_bar
from fairseq.options import add_distributed_training_args

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.validate')


def main(args, override_args=None):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    use_fp16 = args.fp16
    use_cuda = torch.cuda.is_available() and not args.cpu

    if override_args is not None:
        try:
            override_args = override_args['override_args']
        except TypeError:
            override_args = override_args
        overrides = vars(override_args)
        overrides.update(eval(getattr(override_args, 'model_overrides', '{}')))
    else:
        overrides = None

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, model_args, task = checkpoint_utils.load_model_ensemble_and_task(
        [args.path],
        arg_overrides=overrides,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )
    model = models[0]

    # Move models to GPU
    for model in models:
        if use_fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Print args
    logger.info(model_args)

    # Build criterion
    criterion = task.build_criterion(model_args)
    criterion.eval()

    for subset in args.valid_subset.split(','):
        try:
            task.load_dataset(subset, combine=False, epoch=1)
            dataset = task.dataset(subset)
        except KeyError:
            raise Exception('Cannot find dataset: ' + subset)

        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=dataset,
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                *[m.max_positions() for m in models],
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank
        ).next_epoch_itr(shuffle=False)

        progress = progress_bar.progress_bar(
            itr,
            log_format=args.log_format,
            log_interval=args.log_interval,
            prefix=f"valid on '{subset}' subset",
            default_log_format=('tqdm' if not args.no_progress_bar else 'simple'),
        )

        log_outputs = []
        for i, sample in enumerate(progress):
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(
                lambda t: t.half() if t.dtype is torch.float32 else t, sample) if use_fp16 else sample
            try:
                _loss, _sample_size, log_output = task.valid_step(sample, model, criterion)
                progress.log(log_output, step=i)
                log_outputs.append(log_output)
            
            except TypeError:
                break

        with metrics.aggregate() as agg:
            task.reduce_metrics(log_outputs, criterion)
            log_output = agg.get_smoothed_values()
        

        # summarize all the gpus
        if args.distributed_world_size > 1:
            all_log_output = list(zip(*distributed_utils.all_gather_list([log_output])))[0]
            log_output = {
                key: np.mean([log[key] for log in all_log_output])
                for key in all_log_output[0]
            }

        progress.print(log_output, tag=subset, step=i)
       


def cli_main():
    parser = options.get_validation_parser()
    add_distributed_training_args(parser)
    args = options.parse_args_and_arch(parser)

    # only override args that are explicitly given on the command line
    override_parser = options.get_validation_parser()
    add_distributed_training_args(override_parser)
    override_args = options.parse_args_and_arch(override_parser, suppress_defaults=True)

    distributed_utils.call_main(args, main, override_args=override_args)


if __name__ == '__main__':
    cli_main()
