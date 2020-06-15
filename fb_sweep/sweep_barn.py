
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


@register_grid("geo_barn")
def get_ignatius_grid(args):
    gen_args = {
        'render_resolution': '270x480',
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_up_vector': "(0,1,0)",
        'render_path_args': "{'radius': 5.0, 'h': 0, 'axis': 'y', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', '1080x1920', save_dir_key=lambda val: f'{val}'),
        hyperparam('--valid-view-resolution', '540x960'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--train-views', "0..410"),
        hyperparam('--valid-views', "0..320:8"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/barn_srn/meshvoxel0.4.txt"),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/final/tanksandtemple_barn/bbvoxel0.32.txt"),
        # hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--quantized-embed-dim', 32, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--add-pos-embed', 6, save_dir_key=lambda val: f'addpos{val}'),
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 60),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        hyperparam('--transparent-background', "1,1,1", save_dir_key=lambda val: f'bg'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--parallel-sampling', True, binary_flag=True),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,25000"),
        hyperparam('--total-num-embedding', 80000, save_dir_key=lambda val: '80k'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),

        # evaluation with rendering
        hyperparam('--rendering-every-steps', 5000),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        # hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.8),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 64, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--alpha-weight', 1.0, save_dir_key=lambda val: f'alpha{val}'),
        # hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 120000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 250),
        hyperparam('--max-update', 120000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', "/checkpoint/jgu/space/neuralrendering/debug_ignatius/geo_ignatiusp.fp16.single.540x960.s1.v2.geo_nerf.emb384.ss0.1.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint_11_50000.pt")
    # ]
    return hyperparams

@register_grid("geo_barn_final")
def get_barn_final_grid(args):
    gen_args = {
        'render_resolution': '270x480',
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_up_vector': "(0,1,0)",
        'render_path_args': "{'radius': 5.0, 'h': 0, 'axis': 'y', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    hyperparams = [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', '1080x1920', save_dir_key=lambda val: f'{val}'),
        hyperparam('--valid-view-resolution', '540x960'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--train-views', "0..336"),
        hyperparam('--valid-views', "336:384"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/barn_srn/meshvoxel0.4.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/final/tanksandtemple_barnv2/bbvoxel0.32.txt"),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/final/tanksandtemple_barn/new_bbvoxel0.6.txt"),
        hyperparam('--quantized-embed-dim', 32, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--add-pos-embed', 6, save_dir_key=lambda val: f'addpos{val}'),
        hyperparam('--raymarching-stepsize', 0.08, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.6, save_dir_key=lambda val: f'v{val}'),
        # hyperparam('--raymarching-stepsize', 0.04, save_dir_key=lambda val: f'ss{val}'),
        # hyperparam('--voxel-size', 0.32, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 66),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 256, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        hyperparam('--transparent-background', "1,1,1", save_dir_key=lambda val: f'bg'),
        hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--parallel-sampling', True, binary_flag=True),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        hyperparam('--pruning-every-steps', 2500, save_dir_key=lambda val: f'prune{val}'),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        hyperparam('--half-voxel-size-at', "5000,25000,75000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,25000,75000"),
        hyperparam('--total-num-embedding', 80000, save_dir_key=lambda val: '80k'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),

        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 5000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        # hyperparam('--no-load-binary', binary_flag=True),
        hyperparam('--load-mask', binary_flag=True, save_dir_key=lambda val: 'm'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p{val}'),
        hyperparam('--sampling-on-mask', 0.8),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 64, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--alpha-weight', 1.0, save_dir_key=lambda val: f'alpha{val}'),
        # hyperparam('--entropy-weight', 0.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--end-learning-rate', 0.0001),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 250),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', "/checkpoint/jgu/space/neuralrendering/debug_ignatius/geo_ignatiusp.fp16.single.540x960.s1.v2.geo_nerf.emb384.ss0.1.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint_11_50000.pt")
    # ]
    return hyperparams


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
