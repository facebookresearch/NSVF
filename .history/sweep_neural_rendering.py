#!/usr/bin/env python

import fb_sweep.sweep as sweep
from fb_sweep.sweep import hyperparam


GRID_REGISTRY = {}


def register_grid(name):
    def register_grid_cls(cls):
        if name in GRID_REGISTRY:
            raise ValueError('Cannot register duplicate grid ({})'.format(name))
        GRID_REGISTRY[name] = cls
        return cls
    return register_grid_cls


@register_grid("srn")
def get_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--load-depth', binary_flag=True),
        hyperparam('--load-mask', binary_flag=True),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'srn_base', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_start")
def get_start_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        # hyperparam('--load-depth', binary_flag=True),
        hyperparam('--load-mask', binary_flag=True),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'srn_start', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    # param = get_grid(args)
    # param += [
    #     hyperparam('--arch', 'srn_start', save_dir_key=lambda val: val),
        
    # ]
    # return param


@register_grid("srn_debug2")
def get_debug2_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--load-mask', binary_flag=True),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'srn_base', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_debug")
def get_debug_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'pointnet2_srn', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 1000, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_debug4")
def get_debug4_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 4, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),
        hyperparam('--sdf-scale', 1.0, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'geo_srn1', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.08, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_debug_seq")
def get_seq_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 2, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 1),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),
        hyperparam('--max-valid-view', 1),

        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'pointnet2_srn', save_dir_key=lambda val: val),
        # hyperparam('--arch', 'transformer_srn', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 1000, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        # hyperparam('--pointnet2-upsample512', binary_flag=True, save_dir_key=lambda val: 'up512'),
        
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 400),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_debug_seq2")
def get_seq2_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 2, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 1),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),
        hyperparam('--max-valid-view', 1),

        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'transformer_srn', save_dir_key=lambda val: val),
        hyperparam('--ball-radius',1000, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', [True], binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--subsampling-points', [256], save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--encoder-attention-heads', [8], save_dir_key=lambda val: f'head{val}'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 400),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_debug3")
def get_debug3_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'transformer_srn', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_shapenet")
def get_shapenet_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'srn_base', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_shapenet_geo")
def get_shapnetgeo_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 4, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'geo_srn1_dev', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--march-with-bbox', True, binary_flag=True, save_dir_key=lambda val: 'mb'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_shapenet_geo2")
def get_shapnetgeo2_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 7, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 10, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-2),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'dev_srn1', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        
        # hyperparam('--march-with-bbox', True, binary_flag=True, save_dir_key=lambda val: 'mb'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_shapenet_dev")
def get_shapnetdev_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 5, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-2),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        hyperparam('--freespace-weight', 2.0, save_dir_key=lambda val: f'fs{val}'),
        hyperparam('--occupancy-weight', 1.0, save_dir_key=lambda val: f'oc{val}'),
        hyperparam('--no-loss-if-predicted', True, binary_flag=True, 
                    save_dir_key=lambda val: 'nlip'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'dev_srn1', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.125, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--march-with-ball', True, binary_flag=True, save_dir_key=lambda val: 'mball'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),

        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--clip-norm', 80.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_lego_dev")
def get_legodev_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-train-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 4, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-2),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        hyperparam('--freespace-weight', 1000.0, save_dir_key=lambda val: f'fs{val}'),
        hyperparam('--occupancy-weight', 1000.0, save_dir_key=lambda val: f'oc{val}'),
        hyperparam('--no-loss-if-predicted', True, binary_flag=True, 
                    save_dir_key=lambda val: 'nlip'),
        hyperparam('--max-margin-freespace', 0.3, save_dir_key=lambda val: f'freemargin{val}'),
        hyperparam('--no-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'nobg'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--transformer-pos-embed', True, binary_flag=True),
        hyperparam('--transformer-input-shuffle', True, binary_flag=True),
        hyperparam('--subsampling-points', 512, save_dir_key=lambda val: f'sub{val}'),
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        
        
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'dev_srn1', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'ball{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--march-with-ball', True, binary_flag=True, save_dir_key=lambda val: 'mball'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),


        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),

        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_lego_dev2")
def get_legodev2_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 3),
        hyperparam('--max-train-view', 100),
        hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 4, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 20.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-2),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        hyperparam('--freespace-weight', 1000.0, save_dir_key=lambda val: f'fs{val}'),
        hyperparam('--occupancy-weight', 1000.0, save_dir_key=lambda val: f'oc{val}'),
        hyperparam('--no-loss-if-predicted', True, binary_flag=True, 
                    save_dir_key=lambda val: 'nlip'),
        hyperparam('--max-margin-freespace', 0.1, save_dir_key=lambda val: f'freemargin{val}'),
        hyperparam('--no-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'nobg'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),

        # hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        
        
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'dev_srn3', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'ball{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--max-hits', 32),
        # hyperparam('--straight-through-min', True, binary_flag=True, save_dir_key=lambda val: 'st'),
        # hyperparam('--march-with-ball', True, binary_flag=True, save_dir_key=lambda val: 'mball'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),


        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),

        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--clip-norm', 100.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_lego_dev3")
def get_legodev3_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 3),
        hyperparam('--max-train-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 4, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-2),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        hyperparam('--freespace-weight', 1000.0, save_dir_key=lambda val: f'fs{val}'),
        hyperparam('--occupancy-weight', 1000.0, save_dir_key=lambda val: f'oc{val}'),
        hyperparam('--no-loss-if-predicted', True, binary_flag=True, 
                    save_dir_key=lambda val: 'nlip'),
        hyperparam('--max-margin-freespace', 0.3, save_dir_key=lambda val: f'freemargin{val}'),
        hyperparam('--no-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'nobg'),
        
        # specific arguments
        hyperparam('--lstm-sdf', True, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 32),
        hyperparam('--num-layer-features', 6, save_dir_key=lambda val: f'{val}lys'),

        # hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        
        
        hyperparam('--sdf-scale', 0.1, save_dir_key=lambda val: f'sdf{val}'),
        hyperparam('--arch', 'dev_srn3', save_dir_key=lambda val: val),
        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        # hyperparam('--march-with-ball', True, binary_flag=True, save_dir_key=lambda val: 'mball'),
        # hyperparam('--background-feature', True, binary_flag=True, save_dir_key=lambda val: "bgf"),


        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),

        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        # hyperparam('--clip-norm', 100.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.001, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("srn_lego")
def get_lego_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.8, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'srn_base', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        # hyperparam('--dropout', 0.3, save_dir_key=lambda val: f'drop{val}'),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
