
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


@register_grid("geo_scannet2")
def get_birds_grid(args):
    # gen_args = {
    #     'render_resolution': '484x648',
    #     'render_output_types': ["hit", "rgb", "depth", "normal"],
    #     'render_up_vector': "(0,0,1)",
    #     'render_path_args': "{'radius': 3.0, 'h': -1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    # }
    # gen_args = json.dumps(gen_args)

    hyperparams = [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', '480x640', save_dir_key=lambda val: f'{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--train-views', "0..1710"),
        hyperparam('--valid-views', "0..1600:40"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0049_00/data/bbvoxel0.1.txt"),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0049_00/data/bbvoxel0.4.txt"),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 100),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),
        hyperparam('--total-num-embedding', 80000, save_dir_key=lambda val: '80k'),

        # dynamic pruning
        hyperparam('--pruning-every-steps', 2500),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        
        # reducing voxels
        hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,25000"),

        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 1000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p2048'),
        hyperparam('--sampling-at-center', 0.95, save_dir_key=lambda val: f'sc{val}'),
        # hyperparam('--sampling-on-mask', 0.8),
        # hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 256, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.0, save_dir_key=lambda val: f'depth{val}'),
        hyperparam('--entropy-weight', 10.0, save_dir_key=lambda val: f'ent{val}'),
        # hyperparam('--soft-depth-loss', binary_flag=True),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

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
        hyperparam('--save-interval-updates', 500),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 10),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', 
    #     '/checkpoint/jgu/space/neuralrendering/debug_scannetv2/scene0049_00v2.single.480x640.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.60k.th0.5.dyvox.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint_last.pt'),
    #     hyperparam('--reset-optimizer', binary_flag=True, save_dir_key=lambda val: 'reset'),
    # ]
    return hyperparams


@register_grid("geo_scannet2_reset")
def get_reset_grid(args):
    # gen_args = {
    #     'render_resolution': '484x648',
    #     'render_output_types': ["hit", "rgb", "depth", "normal"],
    #     'render_up_vector': "(0,0,1)",
    #     'render_path_args': "{'radius': 3.0, 'h': -1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    # }
    # gen_args = json.dumps(gen_args)

    hyperparams = [
        # hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "single"),
        
        # task level
        hyperparam('--view-resolution', '480x640', save_dir_key=lambda val: f'{val}'),
        hyperparam('--max-sentences', 1, save_dir_key=lambda val: f's{val}'),
        hyperparam('--view-per-batch', 4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--train-views', "0..1710"),
        hyperparam('--valid-views', "0..1600:40"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0049_00/data/bbvoxel0.1.txt"),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0049_00/data/bbvoxel0.4.txt"),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 100),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),
        hyperparam('--total-num-embedding', 80000, save_dir_key=lambda val: '80k'),

        # dynamic pruning
        hyperparam('--pruning-every-steps', 2500),
        hyperparam('--pruning-th', 0.5, save_dir_key=lambda val: f'th{val}'),
        
        # reducing voxels
        hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "5000,25000"),

        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 1000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p2048'),
        hyperparam('--sampling-at-center', 0.95, save_dir_key=lambda val: f'sc{val}'),
        # hyperparam('--sampling-on-mask', 0.8),
        # hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 256, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.0, save_dir_key=lambda val: f'depth{val}'),
        hyperparam('--entropy-weight', 10.0, save_dir_key=lambda val: f'ent{val}'),
        # hyperparam('--soft-depth-loss', binary_flag=True),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.00075, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--end-learning-rate', 0.00001),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [20], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 500),
        hyperparam('--max-update', 100000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 10),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    hyperparams += [
        hyperparam('--restore-file', 
        '/checkpoint/jgu/space/neuralrendering/debug_scannetv2/scene0049_00v2.single.480x640.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.60k.th0.5.dyvox.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint_last.pt'),
        hyperparam('--reset-optimizer', binary_flag=True, save_dir_key=lambda val: 'reset'),
    ]
    return hyperparams


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
