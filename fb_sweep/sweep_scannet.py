
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


@register_grid("geo_scannet")
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
        hyperparam('--train-views', "0..3397"),
        hyperparam('--valid-views', "0..3200:50"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/v3_voxel0.1.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/data/bbvoxel0.1.txt"),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/data/bbvoxel0.1_revised.txt"),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.0125, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 60),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),

        # dynamic pruning
        hyperparam('--pruning-every-steps', 2000),
        hyperparam('--pruning-th', 0.5),
        # hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "5000,25000"),

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
        hyperparam('--soft-depth-loss', binary_flag=True),
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
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', 
    #     ' /checkpoint/jgu/space/neuralrendering/debug_scannetv2/scene0024_00v2.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint_6_29000.pt')
    # ]
    return hyperparams


@register_grid("geo_scannet01")
def get_scannet01_grid(args):
    # gen_args = {
    #     'render_resolution': '484x648',
    #     'render_output_types': ["hit", "rgb", "depth", "normal"],
    #     'render_up_vector': "(0,0,1)",
    #     'render_path_args': "{'radius': 3.0, 'h': -1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    # }
    # gen_args = json.dumps(gen_args)

    BBOX = "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/bbvoxel0.4.txt"
    PVOX = "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.4.txt"
    PVOX1 = "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.1.txt"
    PVOX2 = "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/v3_voxel0.1.txt"

    def get_name(f):
        return f.split('/')[-1].split('.txt')[0]

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
        # hyperparam('--train-views', "1200"),
        # hyperparam('--valid-views', "1200"),
        hyperparam('--train-views', "0..3397"),
        hyperparam('--valid-views', "0..3200:50"),

        # model arguments
        hyperparam('--arch', 'geo_nerf'),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.1.txt"),
        # hyperparam('--quantized-voxel-path', "),
        hyperparam('--quantized-voxel-path', [BBOX], save_dir_key=lambda val: get_name(val)),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        # hyperparam('--raymarching-stepsize', 0.0125, save_dir_key=lambda val: f'ss{val}'),
        # hyperparam('--quantized-xyz-embed', binary_flag=True, save_dir_key=lambda val: f'qxyz'),
        hyperparam('--voxel-size', 0.4, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 64),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),

        # dynamic pruning
        hyperparam('--total-num-embedding', 100000, save_dir_key=lambda val: '100k'),
        hyperparam('--pruning-every-steps', 2000),
        hyperparam('--pruning-th', 0.2, save_dir_key=lambda val: 'pt{val}'),
        hyperparam('--half-voxel-size-at', "10000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--reduce-step-size-at', "10000"),

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
        # hyperparam('--depth-weight', 10.0, save_dir_key=lambda val: f'depth{val}'),
        hyperparam('--entropy-weight', 10.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
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
        hyperparam('--save-interval-updates', 1000),
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
    #     "/checkpoint/jgu/space/neuralrendering/debug_scannet0/scannet0024_00_rgbd_new.single.480x640.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.dis.ps.100k.dyvox.d.p2048.chk512.rgb128.0.depth10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/checkpoint10.pt")
    # ]

    return hyperparams


@register_grid("geo_scannet00")
def get_scannet00_grid(args):
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
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--train-views', "1200"),
        hyperparam('--valid-views', "1200"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/v3_voxel0.1.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/initial_voxel0.2.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.1.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.2.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/clean_voxel0.4.txt"),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/scene0024_00/out/bbvoxel0.4.txt"),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--raymarching-stepsize', 0.01, save_dir_key=lambda val: f'ss{val}'),
        # hyperparam('--raymarching-stepsize', 0.0125, save_dir_key=lambda val: f'ss{val}'),
        # hyperparam('--quantized-xyz-embed', binary_flag=True, save_dir_key=lambda val: f'qxyz'),
        hyperparam('--voxel-size', 0.1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 64),
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        # hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),

        # dynamic pruning
        # hyperparam('--total-num-embedding', 100000, save_dir_key=lambda val: '100k'),
        hyperparam('--pruning-every-steps', 2000),
        hyperparam('--pruning-th', 0.5),
        # hyperparam('--half-voxel-size-at', "10000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--reduce-step-size-at', "10000,50000"),

        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 1000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        hyperparam('--load-depth', binary_flag=True, save_dir_key=lambda val: 'd'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p2048'),
        hyperparam('--sampling-at-center', 0.95, save_dir_key=lambda val: f'sc{val}'),
        # hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 256, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--depth-weight', 0.0, save_dir_key=lambda val: f'depth{val}'),
        hyperparam('--entropy-weight', 10.0, save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        # hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
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
        hyperparam('--save-interval-updates', 100),
        hyperparam('--max-update', 10000),
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', 
    #     "/checkpoint/jgu/space/neuralrendering/debug_scannet1/scannet0024_00_rgbdv4.single.480x640.s1.v1.geo_nerf.emb384.ss0.01.v0.1.posemb.sdfh128.raydir.r24.dis.ps.d.p2048.chk512.rgb0.0.depth0.0.ent1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu1/checkpoint_last.pt",
    #     # "/checkpoint/jgu/space/neuralrendering/debug_scannet1/scannet0024_00_rgbdv3.single.480x640.s1.v1.geo_nerf.emb384.ss0.01.v0.1.posemb.sdfh128.raydir.r24.dis.ps.dyvox.d.p2048.chk512.rgb0.0.depth10.0.ent1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu1/checkpoint_last.pt",
    #     save_dir_key=lambda val: f'reload')
    # ]
    return hyperparams


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
