#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random, shlex
import os, sys, subprocess


def launch_cluster(slurm_args, model_args):
    # prepare
    jobname = slurm_args.get('job-name', 'test')
    train_log = slurm_args.get('output', None)
    train_stderr = slurm_args.get('error', None)
    nodes, gpus = slurm_args.get('nodes', 1), slurm_args.get('gpus', 8)
    if not slurm_args.get('local', False):
        assert (train_log is not None) and (train_stderr is not None)

    # parse slurm
    train_cmd = ['python', 'train.py', ]
    train_cmd.extend(['--distributed-world-size', str(nodes * gpus)])
    if nodes > 1:
        train_cmd.extend(['--distributed-port', str(get_random_port())])
    
    train_cmd += model_args

    base_srun_cmd = [
            'srun',
            '--job-name', jobname,
            '--output', train_log,
            '--error', train_stderr,
            '--open-mode', 'append',
            '--unbuffered',
        ]
    srun_cmd = base_srun_cmd + train_cmd
    srun_cmd_str = ' '.join(map(shlex.quote, srun_cmd)) 
    srun_cmd_str = srun_cmd_str + ' &'

    sbatch_cmd = [
                'sbatch',
                '--job-name', jobname,
                '--partition', slurm_args.get('partition', 'learnfair'),
                '--gres', 'gpu:volta:{}'.format(gpus),
                '--nodes', str(nodes),
                '--ntasks-per-node', '1',
                '--cpus-per-task', '48',
                '--output', train_log,
                '--error', train_stderr,
                '--open-mode', 'append',
                '--signal', 'B:USR1@180',
                '--time', slurm_args.get('time', '4320'),
                '--mem', slurm_args.get('mem', '500gb'),
                '--exclusive',
            ]
    if 'constraint' in slurm_args:
        sbatch_cmd += ['-C', slurm_args.get('constraint')]
    if 'comment' in slurm_args:
        sbatch_cmd += ['--comment', slurm_args.get('comment')]
    
    wrapped_cmd = requeue_support() + '\n' + srun_cmd_str + ' \n wait $! \n sleep 610 & \n wait $!'
    sbatch_cmd += ['--wrap', wrapped_cmd]
    sbatch_cmd_str = ' '.join(map(shlex.quote, sbatch_cmd))
    
    # start training
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = '2'
    if env.get('SLURM_ARGS', None) is not None:
        del env['SLURM_ARGS']

    if nodes > 1:
        env['NCCL_SOCKET_IFNAME'] = '^docker0,lo'
        env['NCCL_DEBUG'] = 'INFO'

    if slurm_args.get('dry-run', False):
        print(sbatch_cmd_str)
    
    elif slurm_args.get('local', False):
        assert nodes == 1, 'distributed training cannot be combined with local' 
        if 'CUDA_VISIBLE_DEVICES' not in env:
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(gpus)))
        env['NCCL_DEBUG'] = 'INFO'

        if train_log is not None:
            train_proc = subprocess.Popen(train_cmd, env=env, stdout=subprocess.PIPE)
            tee_proc = subprocess.Popen(['tee', '-a', train_log], stdin=train_proc.stdout)
            train_proc.stdout.close()
            train_proc.wait()
            tee_proc.wait()
        else:
            train_proc = subprocess.Popen(train_cmd, env=env)
            train_proc.wait()
    else:
        with open(train_log, 'a') as train_log_h:
            print(f'running command: {sbatch_cmd_str}\n')
            with subprocess.Popen(sbatch_cmd, stdout=subprocess.PIPE, env=env) as train_proc:
                stdout = train_proc.stdout.read().decode('utf-8')
                print(stdout, file=train_log_h)
                try:
                    job_id = int(stdout.rstrip().split()[-1])
                    return job_id
                except IndexError:
                    return None


def launch(slurm_args, model_args):
    job_id = launch_cluster(slurm_args, model_args)
    if job_id is not None:
        print('Launched {}'.format(job_id))
    else:
        print('Failed.')


def requeue_support():
    return """
        trap_handler () {
           echo "Caught signal: " $1
           # SIGTERM must be bypassed
           if [ "$1" = "TERM" ]; then
               echo "bypass sigterm"
           else
             # Submit a new job to the queue
             echo "Requeuing " $SLURM_JOB_ID
             scontrol requeue $SLURM_JOB_ID
           fi
        }


        # Install signal handler
        trap 'trap_handler USR1' USR1
        trap 'trap_handler TERM' TERM
    """


def get_random_port():
    old_state = random.getstate()
    random.seed()
    port = random.randint(10000, 20000)
    random.setstate(old_state)
    return port
