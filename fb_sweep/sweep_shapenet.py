
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


@register_grid("geo_shapenet")
def get_maria_seq2_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    voxel_name = "/private/home/jgu/data/shapenet/shapenet_chair/render_256/9d36bf414dde2f1a93a28cbb4bfc693b/voxel.txt"

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "single_object_rendering", save_dir_key=lambda val: "seq"),
        
        # task level
        # hyperparam('--view-resolution', 256, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 2, save_dir_key=lambda val: f's4'),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        # hyperparam('--subsample-valid', 100),
        hyperparam('--max-train-view', 50),
        hyperparam('--max-valid-view', 50),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/shapenet_chair/initial_voxel.txt"),
        hyperparam('--quantized-voxel-path', voxel_name),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        # hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 5000),
        hyperparam('--raymarching-stepsize', 0.025, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.25, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        # hyperparam('--background-stop-gradient', True, binary_flag=True, save_dir_key=lambda val: f'bgsg'),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        # hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        # hyperparam('--condition-on-marchsize', binary_flag=True, save_dir_key=lambda val: f'cm'),
        # hyperparam('--half-voxel-size-at', "10000,50000", save_dir_key=lambda val: f'dyvox'),
        # evaluation with rendering
        # hyperparam('--rendering-every-steps', 1000),
        # hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        # hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 16384, save_dir_key=lambda val: f'p16384'),
        # hyperparam('--sampling-on-mask', 0.5, save_dir_key=lambda val: f'smk{val}'),
        # hyperparam('--sampling-on-bbox', binary_flag=True),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--entropy-weight', [10.0], save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0),
        hyperparam('--vgg-weight', 1.0, save_dir_key=lambda val: f'vgg{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 200000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--end-learning-rate', 0.0001),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip{val}'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 1000),
        hyperparam('--max-update', 200000),
        
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_shapenet_seq")
def get_shapenet_seq_grid(args):
    gen_args = {
        'render_resolution': 512,
        'render_output_types': ["target", "hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    voxel_name = "/private/home/jgu/data/srn_data/initial_voxel.txt"

    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "seq"),
        
        # task level
        hyperparam('--view-resolution', 512, save_dir_key=lambda val: f'{val}x{val}'),
        # hyperparam('--view-resolution', 128, save_dir_key=lambda val: f'{val}x{val}'),
        # hyperparam('--max-sentences', 16, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences', 4, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 4),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--subsample-valid', 100),
        hyperparam('--max-train-view', 50),
        hyperparam('--max-valid-view', 50),

        # model arguments
        # hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--arch', 'geo_nerf_transformer', save_dir_key=lambda val: val),

        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/shapenet_chair/initial_voxel.txt"),
        hyperparam('--quantized-voxel-path', voxel_name),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 5000),
        
        # -- transformer params
        # hyperparam('--over-residual', False, binary_flag=True, save_dir_key=lambda val: f'tres'),
        hyperparam('--encoder-attention-heads', 1),
        hyperparam('--encoder-layers', [1], save_dir_key=lambda val: f'enc{val}'),
        hyperparam('--cross-attention-context', binary_flag=True, save_dir_key=lambda val: f'cac'),
    
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.25, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--condition-on-marchsize', binary_flag=True, save_dir_key=lambda val: f'cm'),
        hyperparam('--half-voxel-size-at', "10000,50000", save_dir_key=lambda val: f'dyvox'),
        
        # evaluation with rendering
        hyperparam('--rendering-every-steps', 2500),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        # hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 4096, save_dir_key=lambda val: f'p{val}'),
        # hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.9, save_dir_key=lambda val: f'smk{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 200.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--alpha-weight', 1.0, save_dir_key=lambda val: f'alpha{val}'),
        # hyperparam('--entropy-weight', [10.0], save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 0.0), # , save_dir_key=lambda val: f'latent{val}'),
        hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 200000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--end-learning-rate', 0.0001),
        hyperparam('--clip-norm', 50.0, save_dir_key=lambda val: f'clip0.0'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 1000),
        hyperparam('--max-update', 200000),
        
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_shapenet_seq128")
def get_shapenet_seq128_grid(args):
    gen_args = {
        'render_resolution': 128,
        'render_output_types': ["target", "hit", "rgb", "depth", "normal"],
        'render_path_args': "{'radius': 3.0, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}",
    }
    gen_args = json.dumps(gen_args)

    voxel_name = "/private/home/jgu/data/srn_data/initial_voxel.txt"
    hyperparams = [
        hyperparam('--fp16', save_dir_key=lambda val: 'fp16'),
        # hyperparam('--ddp-backend', 'no_c10d', save_dir_key=lambda val: 'no_c10d'),
        hyperparam('--broadcast-buffers', binary_flag=True), # adding it to broadcast batchnorm (if needed)
        hyperparam('--task', "sequence_object_rendering", save_dir_key=lambda val: "seq"),
        
        # task level
        hyperparam('--view-resolution', 128, save_dir_key=lambda val: f'{val}x{val}'),
        hyperparam('--max-sentences', 16, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 4),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 2),
        hyperparam('--subsample-valid', 10),
        hyperparam('--max-train-view', 50),
        hyperparam('--max-valid-view', 10),

        # model arguments
        # hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam('--arch', 'geo_nerf_transformer', save_dir_key=lambda val: val),

        # hyperparam('--quantized-voxel-path', "/private/home/jgu/data/shapenet/shapenet_chair/initial_voxel.txt"),
        hyperparam('--quantized-voxel-path', voxel_name),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 5000),
        
        # -- transformer params
        # hyperparam('--over-residual', False, binary_flag=True, save_dir_key=lambda val: f'tres'),
        hyperparam('--encoder-attention-heads', 1),
        hyperparam('--encoder-layers', [1], save_dir_key=lambda val: f'enc{val}'),
        hyperparam('--cross-attention-context', binary_flag=True, save_dir_key=lambda val: f'cac'),
    
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.25, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 48),
        
        hyperparam('--pos-embed', True, binary_flag=True, save_dir_key=lambda val: f'posemb'),
        hyperparam('--hidden-sdf', 128, save_dir_key=lambda val: f'sdfh{val}'),
        # hyperparam('--use-raydir', True, binary_flag=True, save_dir_key=lambda val: 'raydir'),
        # hyperparam('--raydir-features', 24, save_dir_key=lambda val: f'r{val}'),
        
        # hyperparam('--raypos-features', 0, save_dir_key=lambda val: f'pos{val}'),
        # hyperparam('--saperate-specular', True, binary_flag=True, save_dir_key=lambda val: 'spec'),
        # hyperparam('--specular-dropout', 0.5, save_dir_key=lambda val: f'sd{val}'),
        hyperparam('--transparent-background', 1.0, save_dir_key=lambda val: f'bg{val}'),
        hyperparam('--background-stop-gradient', True, binary_flag=True),
        hyperparam('--discrete-regularization', True, binary_flag=True, save_dir_key=lambda val: f'dis'),
        hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),

        # dynamic pruning
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--condition-on-marchsize', binary_flag=True, save_dir_key=lambda val: f'cm'),
        hyperparam('--half-voxel-size-at', "10000,50000", save_dir_key=lambda val: f'dyvox'),
        
        # evaluation with rendering
        hyperparam('--rendering-every-steps', 2500),
        hyperparam('--rendering-args', gen_args),

        # dataset arguments
        hyperparam('--no-preload', binary_flag=True),
        # hyperparam('--load-point', binary_flag=True, save_dir_key=lambda val: 'p'),

        # training arguments
        hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p{val}'),
        # hyperparam('--pixel-per-view', 2048, save_dir_key=lambda val: f'p16384'),
        hyperparam('--sampling-on-mask', 0.9, save_dir_key=lambda val: f'smk{val}'),
        hyperparam('--sampling-on-bbox', binary_flag=True),
        # hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
        hyperparam('--chunk-size', 512, save_dir_key=lambda val: f'chk512'),
        hyperparam('--inner-chunking', False, binary_flag=True),        
        
        hyperparam('--rgb-weight', 128.0, save_dir_key=lambda val: f'rgb{val}'),
        hyperparam('--alpha-weight', 1.0, save_dir_key=lambda val: f'alpha{val}'),
        # hyperparam('--entropy-weight', [10.0], save_dir_key=lambda val: f'ent{val}'),
        hyperparam('--reg-weight', 1.0, save_dir_key=lambda val: f'latent{val}'),
        hyperparam('--vgg-weight', 0.0, save_dir_key=lambda val: f'vgg{val}'),
        # hyperparam('--vgg-level', 3, save_dir_key=lambda val: f'l{val}'),

        hyperparam('--optimizer', 'adam', save_dir_key=lambda val: val),
        hyperparam('--adam-betas', '(0.9, 0.999)'),
        # hyperparam('--lr-scheduler', 'fixed', save_dir_key=lambda val: f"lr_{val}"),
        hyperparam('--lr-scheduler', 'polynomial_decay', save_dir_key=lambda val: f'lr_poly'),
        hyperparam('--total-num-update', 200000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--lr', 0.001, save_dir_key=lambda val: f'lr{val}'),
        hyperparam('--end-learning-rate', 0.0001),
        hyperparam('--clip-norm', 0.0, save_dir_key=lambda val: f'clip0.0'),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0, save_dir_key=lambda val: f'wd{val}'),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 1000),
        hyperparam('--max-update', 200000),
        
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams



def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
