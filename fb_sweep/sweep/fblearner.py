from collections import OrderedDict
import itertools
import json
import os
import random
import shlex
import shutil
import subprocess
import tempfile


def main(get_grid, postprocess_hyperparams, args):
    # compute all possible hyperparameter configurations
    grid = get_grid(args)
    grid_product = list(itertools.product(*[hp.values for hp in grid]))

    # randomly shuffle configurations
    random.seed(args.seed)
    random.shuffle(grid_product)

    sweep_config = {}
    save_dirs = []
    for i, hp_values in enumerate(grid_product):
        config = OrderedDict()
        for hp, value in zip(grid, hp_values):
            config[hp.name] = hp
            config[hp.name].current_value = value

        # postprocess hyperparams
        postprocess_hyperparams(args, config)

        # setup training
        x = setup_train(args, config)
        if x is not None:
            sweep_config[x['train_log_path']] = x['cmd_args']
            save_dirs.append(x['save_dir'])

        if i == args.num_trials - 1:
            break

    if len(save_dirs) == 0:
        return

    with tempfile.NamedTemporaryFile('w') as h:
        config = {
            "cmd_args": [],
            "num_nodes": args.num_nodes,
            "num_gpus_per_node": args.num_gpus,
            "fp16": any(
                "--fp16" in cmd_args or "--memory-efficient-fp16" in cmd_args
                for cmd_args in sweep_config.values()
            ),
            "sweep_config": sweep_config,
        }
        h.write(json.dumps(config))
        h.flush()

        # build flow command
        prefix = args.prefix.rstrip('_')
        num_total_gpus = args.num_nodes * args.num_gpus
        flow_cmd = [
            '/usr/local/bin/flow-cli',
            'canary',
            #'--py-version', '>=3',
            '--mode', 'opt',
            '--entitlement', str(args.entitlement),
            '--run-as-secure-group', args.run_as_secure_group,
            '--parameters-file', str(h.name),
            '--name', f"{prefix}.ngpu{num_total_gpus}",
            'fairseq.train.train_workflow'
            # TODO put stuff in --notes, e.g., repro command
        ]
        cmd = ' '.join(map(shlex.quote, flow_cmd))

        if args.dry_run:
            print('| dry-run: start remote training')
            print(f'| dry-run: - run command: {cmd}')
        else:
            subprocess.Popen(
                flow_cmd,
                cwd=os.path.join(
                    '/data/users', os.environ['USER'], 'fbsource/fbcode',
                ),
            ).wait()


def setup_train(args, config):
    def dry_run(msg):
        if args.dry_run:
            print(f'| dry-run:  {msg}')
        return args.dry_run

    # compute save_dir
    save_dir_key = '.'.join(filter(
        lambda save_dir_key: save_dir_key is not None,
        [hp.get_save_dir_key() for hp in config.values()]
    ))
    save_dir_key = save_dir_key.replace(",", "_")
    num_total_gpus = args.num_nodes * args.num_gpus
    save_dir = os.path.join(args.checkpoints_dir, f'{args.prefix}.{save_dir_key}.ngpu{num_total_gpus}')

    # create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        if not dry_run(f'create directory: {save_dir}'):
            os.makedirs(save_dir)
            os.chmod(save_dir, 0o777)

        # copy baseline model
        checkpoint_last = os.path.join(save_dir, 'checkpoint_last.pt')
        if args.baseline_model and not os.path.exists(checkpoint_last) and \
                not dry_run(f'initialize with baseline model: {args.baseline_model}'):
            if not os.path.exists(args.baseline_model):
                raise FileNotFoundError(f'Cannot find baseline model: {args.baseline_model}')
            shutil.copyfile(args.baseline_model, checkpoint_last)

    # TODO make this work
    # check for whether the run failed
    #if has_finished(save_dir):
    #    if args.resume_finished:
    #        dry_run(f'restart previously finished run: {save_dir}')
    #    else:
    #        print(f'skip finished run (override with --resume-finished): {save_dir}')
    #        return
    #elif has_failed(save_dir):
    #    if args.resume_failed:
    #        dry_run(f'resume failed run: {save_dir}')
    #    else:
    #        print(f'skip failed run (override with --resume-failed): {save_dir}')
    #        return
    #elif has_started(save_dir):
    if has_started(save_dir):
        print(f'skip in progress run: {save_dir}')
        return

    # generate train command
    cmd_args = [args.data, '--save-dir', save_dir]
    for hp in config.values():
        cmd_args.extend(map(str, hp.get_cli_args()))
    if args.dry_run:
        cmd_args_str = ' '.join(cmd_args)
        dry_run(f'train command: train.par {cmd_args_str}')

    # initialize train log
    train_log = os.path.join(save_dir, 'train.log')
    if not dry_run(f'create train.log at: {train_log}'):
        with open(train_log, 'a') as train_log_h:
            train_log_h.write('')
        os.chmod(train_log, 0o777)

    return {
        'cmd_args': cmd_args,
        'save_dir': save_dir,
        'save_dir_key': save_dir_key,
        'train_log_path': train_log,
    }


#def has_finished(save_dir):
#    train_log = os.path.join(save_dir, 'train.log')
#    if not os.path.exists(train_log):
#        return False
#    with open(train_log, 'r') as h:
#        lines = h.readlines()
#        if len(lines) == 0:
#            return False
#        if 'done training' in lines[-1]:
#            return True
#    return False
#
#
#def has_failed(save_dir):
#    if not os.path.exists(save_dir):
#        return False
#
#    # find max job id
#    job_ids = []
#    for fn in os.listdir(save_dir):
#        if fn.startswith('train.stderr.'):
#            job_ids.append(int(fn.split('.')[-1]))
#    if len(job_ids) == 0:
#        return False
#    max_job_id = max(job_ids)
#
#    def _has_failed(stderr_fn):
#        with open(stderr_fn, 'r') as h:
#            for line in h:
#                if len(line.strip()) > 0:
#                    # assume that any output in stderr indicates an error
#                    return True
#        return False
#
#    return _has_failed(os.path.join(save_dir, f'train.stderr.{max_job_id}'))


def has_started(save_dir):
    train_log = os.path.join(save_dir, 'train.log')
    if not os.path.exists(train_log):
        return False
    return True
