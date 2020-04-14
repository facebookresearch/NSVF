
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


@register_grid("geo_maria_t")
def get_maria_trans_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 50),
        
        # model arguments
        hyperparam('--arch', 'geo_nerf_transformer', save_dir_key=lambda val: val),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/maria/initial_voxel.txt"),
        hyperparam('--raymarching-stepsize', 0.025, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.2, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        # hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        # hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        # hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "5000,50000"),

        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 1000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-load-binary', binary_flag=True),
        #hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.6),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

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


@register_grid("geo_maria")
def get_maria_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 2, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--max-train-view', 50),
        
        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/maria/initial_voxel.txt"),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.025, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.2, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        # hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        # hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        # hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "5000,50000"),

        # evaluation with rendering
        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-load-binary', binary_flag=True),
        #hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.6),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

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


@register_grid("geo_maria_seq")
def get_maria_seq_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "seq"),
        
        # task level
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--max-train-view', 50),
        
        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/maria/initial_voxel.txt"),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 200),
        hyperparam('--raymarching-stepsize', 0.025, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.2, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        # hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        # hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        # hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "5000,50000"),

        # evaluation with rendering
        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        # hyperparam('--no-load-binary', binary_flag=True),
        # hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.6),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

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


@register_grid("geo_maria_seq_reload")
def get_maria_seqr_grid(args):
    hyperparams = get_maria_seq_grid(args)
    hyperparams += [
        hyperparam("--restore-file", 
            "/checkpoint/jgu/space/neuralrendering/debug_new_mariaseq/maria_seq_small_PRUNE5.fp16.seq.512x512.s1.v1.geo_nerf.emb378.id.ss0.025.v0.2.posemb.sdfh128.dis.p16384.chk512.rgb200.0.ent0.0.vgg1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.reload.pruning.p1.0.dyvox.ngpu8/checkpoint_last.pt",
            save_dir_key=lambda val: f'reload'),
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--pruning-weight', 1.0, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--half-voxel-size-at',  "5000,250000,500000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,250000,500000"),
    ]
    return hyperparams

@register_grid("geo_maria_seq2")
def get_maria_seq2_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "seq"),
        
        # task level
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 2, save_dir_key=lambda val: f's4'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--max-train-view', 50),
        
        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/maria/initial_voxel.txt"),
        hyperparam('--quantized-embed-dim', 378, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 200),
        hyperparam('--raymarching-stepsize', 0.025, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.2, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        # hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        # hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        # hyperparam('--half-voxel-size-at', "5000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "5000,50000"),

        # evaluation with rendering
        hyperparam('--rendering-every-steps', 1000),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        # hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p16384'),
        # hyperparam('--sampling-on-mask', 0.8, save_dir_key=lambda val: f'smk{val}'),
        # hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

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
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 1000),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_maria_seq2_dyn")
def get_maria_seq2r_grid(args):
    hyperparams = get_maria_seq2_grid(args)
    hyperparams += [
        # hyperparam("--restore-file", 
        #     "/checkpoint/jgu/space/neuralrendering/debug_new_mariaseq/maria_seq_small_PRUNE5.fp16.seq.512x512.s1.v1.geo_nerf.emb378.id.ss0.025.v0.2.posemb.sdfh128.dis.p16384.chk512.rgb200.0.ent0.0.vgg1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.reload.pruning.p1.0.dyvox.ngpu8/checkpoint_last.pt",
        #     save_dir_key=lambda val: f'reload'),
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--pruning-weight', 1.0, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--half-voxel-size-at',  "10000,40000,80000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "10000,40000,80000"),
    ]
    return hyperparams


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
