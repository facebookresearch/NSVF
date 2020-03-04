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
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
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
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),
        # hyperparam('--load-depth', binary_flag=True),
        
        hyperparam('--load-mask', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'geo'),
        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.08, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--arch', 'geosrn_simple', save_dir_key=lambda val: val),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        hyperparam('--depth-weight-decay', "(0,30000)", save_dir_key=lambda val: 'dwd'),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: 'poly'),
        hyperparam('--total-num-update', 100000),
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

    #     hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'geo'),
    #     hyperparam('--arch', 'geosrn_simple', save_dir_key=lambda val: val),
        
    # ]
    # return param


@register_grid("srn_shapenet")
def get_shapenet_grid(args):
    return [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 5, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--pixel-per-view', 30000, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.0, save_dir_key=lambda val: f'mask{val}'),
        #hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-steps', 10, save_dir_key=lambda val: f'march{val}'),

        hyperparam('--rgb-weight', 200, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 1.0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 1e-3),
        hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),

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
