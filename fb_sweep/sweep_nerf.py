
#!/usr/bin/env python

import json
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


@register_grid("geo_nerf_lego")
def get_lego_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        

        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
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


@register_grid("geo_nerf_lego_dev")
def get_legodev_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-voxel-vertex', True, binary_flag=True),     
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("geo_nerf_hotdog")
def get_hotdog_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.005, save_dir_key=lambda val: f'stepsize{val}'),
        # hyperparam('--raymarching-stepsize', 0.002, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-voxel-vertex', True, binary_flag=True),     
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.06, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--ball-radius', 0.03, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("geo_nerf_hotdog_reload")
def get_hotdog_reload_grid(args):

    gen_args = {
        'render_resolution': 400,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 0.8, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,50000"),

        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--raymarching-stepsize', 0.0075, save_dir_key=lambda val: f'ss{val}'),

        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--density-weight', 0.1, save_dir_key=lambda val: f'den{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--random-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'rbl'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_dyn', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--nerf-pos', True, binary_flag=True, save_dir_key=lambda val: f'np'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.06, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 36, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    # hyperparams += [
    #     hyperparam('--restore-file', 
    #         "/checkpoint/jgu//space/neuralrendering/" + 
    #         "debug_nerf_hotdog3/hotdog2ver4.fp16.single.400x400.s1.v1.prune2500.th0.5.dyvox.p16384.ss0.005.chk512.p.rgb200.0.vgg1.0.geo_nerf_dyn.sdfh128.posemb.emb378.v0.06.bg1.0.bgsg.dis.aabb.raydir.r24.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/" + 
    #         "checkpoint_last.pt", save_dir_key=lambda val: "re"),
    #     # hyperparam('--save-interval-updates', 1),
    # ]
    return hyperparams


@register_grid("geo_new_hotdog")
def get_newhotdog_grid(args):
    gen_args = {
        'render_resolution': 400,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.5, 'h': 1.0, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        hyperparam('--max-valid-view', 100),
        
        # hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        # hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        # hyperparam('--half-voxel-size-at', "10000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "10000,50000"),

        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.8),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--raymarching-stepsize', 0.02, save_dir_key=lambda val: f'ss{val}'),

        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--density-weight', 0.1, save_dir_key=lambda val: f'den{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--random-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'rbl'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_dyn', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.16, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--ball-radius', 0.03, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_new_hotdog2")
def get_newhotdog2_grid(args):
    gen_args = {
        'render_resolution': 400,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.5, 'h': 1.0, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        hyperparam('--max-valid-view', 100),
        
        hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,50000"),

        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--raymarching-stepsize', 0.04, save_dir_key=lambda val: f'ss{val}'),

        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--density-weight', 0.1, save_dir_key=lambda val: f'den{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--random-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'rbl'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_dyn', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.32, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--ball-radius', 0.03, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_nerf_lego_dev2")
def get_legodev2_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--outer-chunk-size', 400*100),
        hyperparam('--inner-chunking', False, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-voxel-vertex', True, binary_flag=True),     
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        hyperparam('--expectation', "features", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("geo_nerf_lego_dev3")
def get_legodev3_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--outer-chunk-size', 400*100),
        hyperparam('--inner-chunking', False, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 36),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-voxel-vertex', True, binary_flag=True),     
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        hyperparam('--expectation', "features", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--save-interval', 100000),
        hyperparam('--max-update', 100000),
        hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]


@register_grid("geo_new_lego")
def get_newlego_grid(args):
    gen_args = {
        'render_resolution': 400,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.5, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 400, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        hyperparam('--max-valid-view', 100),
        
        hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,50000"),

        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        hyperparam('--pixel-per-view', 10000, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.5),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--raymarching-stepsize', 0.02, save_dir_key=lambda val: f'ss{val}'),

        hyperparam('--chunk-size', 256, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--density-weight', 0.1, save_dir_key=lambda val: f'den{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--random-background-loss', True, binary_flag=True, save_dir_key=lambda val: 'rbl'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_dyn', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--raypos-features', 144, save_dir_key=lambda val: f'pos{val}'),

        hyperparam('--ball-radius', 0.24, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--ball-radius', 0.03, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--expectation', "depth", save_dir_key=lambda val: f'e_{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),
        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 200),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_nerf_maria")
def get_maria_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 2, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 4),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.8, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.005, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        

        hyperparam('--ball-radius', 0.07, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
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


@register_grid("geo_nerf_t033")
def get_t033_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'?x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.6, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        # hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        

        hyperparam('--ball-radius', 0.14, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
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


@register_grid("geo_nerf_vlad")
def get_vlad_grid(args):
    return [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        hyperparam('--view-resolution', 514, save_dir_key=lambda val: f'?x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 100),
        # hyperparam('--max-valid-view', 100),
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.6, save_dir_key=lambda val: f'mask{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True, save_dir_key=lambda val: 'bbox'),
        hyperparam('--raymarching-stepsize', 0.005, save_dir_key=lambda val: f'stepsize{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk{val}'),
        hyperparam('--inner-chunking', True, binary_flag=True),

        hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),
        # hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        # hyperparam('--depth-weight', 0, save_dir_key=lambda val: f'dep{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--freespace-weight', 0.0, save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--occupancy-weight', 0.0, save_dir_key=lambda val: f'oc{val}'),

        # specific arguments
        hyperparam('--arch', 'geo_nerf_tri', save_dir_key=lambda val: val),
        hyperparam('--lstm-sdf', False, binary_flag=True, save_dir_key=lambda val: 'lstm'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--quantized-subsampling-points', 0, save_dir_key=lambda val: f'sub{val}'),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-voxel-vertex', True, binary_flag=True),     
        # hyperparam('--quantized-input-shuffle', False, binary_flag=True),  
        # hyperparam('--bounded', True, binary_flag=True, save_dir_key=lambda val: 'bd'),        

        hyperparam('--ball-radius', 0.1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--relative-position', binary_flag=True, save_dir_key=lambda val: 'rel'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--intersection-type', 'aabb', save_dir_key=lambda val: f'{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--sigmoid-activation', True, binary_flag=True, save_dir_key=lambda val: 'sig'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
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


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
