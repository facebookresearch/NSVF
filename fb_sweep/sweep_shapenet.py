
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
        'render_resolution': '128x128',
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
        hyperparam('--view-resolution', '128x128', save_dir_key=lambda val: f'{val}'),
        hyperparam('--max-sentences', 16, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 8),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--subsample-valid', 10),
        hyperparam('--train-views', "0..50"),
        hyperparam('--valid-views', "0..10"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--arch', 'geo_nerf_transformer', save_dir_key=lambda val: val),

        # -- dynamic embedding params
        # hyperparam('--quantized-pos-embed', binary_flag=True, save_dir_key=lambda val: f'qpos'),
        # hyperparam('--quantized-xyz-embed', binary_flag=True, save_dir_key=lambda val: f'qxyz'),
        # hyperparam('--quantized-context-proj', binary_flag=True, save_dir_key=lambda val: f'cp'),
        # hyperparam('--use-hypernetwork', binary_flag=True, save_dir_key=lambda val: f'hyper'),
        # hyperparam('--normalize-context', binary_flag=True, save_dir_key=lambda val: f'nc')
        hyperparam('--post-context', binary_flag=True, save_dir_key=lambda val: f'hpc'),
        hyperparam('--quantized-voxel-path', voxel_name),
        hyperparam('--quantized-embed-dim', 384, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 5000),
        
        # -- transformer params
        # hyperparam('--over-residual', False, binary_flag=True, save_dir_key=lambda val: f'tres'),
        # hyperparam('--encoder-attention-heads', 1),
        # hyperparam('--encoder-layers', [1], save_dir_key=lambda val: f'enc{val}'),
        # hyperparam('--cross-attention-context', binary_flag=True, save_dir_key=lambda val: f'cac'),
        # hyperparam('--attention-context', binary_flag=True, save_dir_key=lambda val: f'ac'),

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
        hyperparam('--sampling-patch-size', 4, save_dir_key=lambda val: f'patch{val}'),
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
        hyperparam('--clip-norm', 0.0),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.0),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [2], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 500),
        hyperparam('--max-update', 200000),
        
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]
    return hyperparams


@register_grid("geo_shapenet_seq128_fewshot2")
def get_shapenet_seq128fs2_grid(args):
    hyperparams = get_shapenet_seq128_grid(args)
    hyperparams += [
        hyperparam('--train-views', "64", save_dir_key=lambda val: f'fs{val}'),
        hyperparam('--valid-views', "0..251"),
        hyperparam('--subsample-valid', 1),
        # hyperparam('--valid-views', "0,50,100,150,200,250"),
        hyperparam("--restore-file", 
            "/checkpoint/jgu/space/neuralrendering/debug_new_chairs/train_bigbatch.fp16.seq.128x128.s16.v1.geo_nerf.emb384.id.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.p16384.smk0.9.chk512.rgb200.0.alpha1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.clip0.0.wd0.0.seed2.ngpu8/checkpoint_last.pt"),
        
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        hyperparam("--freeze-networks", binary_flag=True),
        hyperparam("--reset-context-embed", binary_flag=True),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--max-update', 100000),
        
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--condition-on-marchsize', binary_flag=True, save_dir_key=lambda val: f'cm'),
        hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--save-interval-updates', 500),
        hyperparam('--reset-optimizer', binary_flag=True),
        hyperparam('--reset-meters', binary_flag=True),
        hyperparam('--reset-dataloader', binary_flag=True),
        hyperparam('--reset-lr-scheduler', binary_flag=True),
    ]
    return hyperparams


@register_grid("geo_shapenet_seq1282")
def get_shapenet_seq1282_grid(args):
    gen_args = {
        'render_resolution': "128x128",
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
        hyperparam('--view-resolution', "128x128", save_dir_key=lambda val: f'{val}'),
        hyperparam('--max-sentences', 16, save_dir_key=lambda val: f's{val}'),
        hyperparam('--max-sentences-valid', 8),
        hyperparam('--view-per-batch', 1, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--valid-view-per-batch', 1),
        hyperparam('--subsample-valid', 10),
        hyperparam('--train-views', "0..50"),
        hyperparam('--valid-views', "0..10"),

        # model arguments
        hyperparam('--arch', 'geo_nerf', save_dir_key=lambda val: val),
        # hyperparam('--arch', 'geo_nerf_transformer', save_dir_key=lambda val: val),

        # -- dynamic embedding params
        # hyperparam('--quantized-pos-embed', binary_flag=True, save_dir_key=lambda val: f'qpos'),
        hyperparam('--quantized-xyz-embed', binary_flag=True, save_dir_key=lambda val: f'qxyz'),
        # hyperparam('--quantized-context-proj', binary_flag=True, save_dir_key=lambda val: f'cp'),
        # hyperparam('--use-hypernetwork', binary_flag=True, save_dir_key=lambda val: f'hyper'),
        # hyperparam('--normalize-context', binary_flag=True, save_dir_key=lambda val: f'nc')
        hyperparam('--post-context', binary_flag=True, save_dir_key=lambda val: f'hpc'),
        hyperparam('--quantized-voxel-path', voxel_name),
        hyperparam('--quantized-embed-dim', 128, save_dir_key=lambda val: f'emb{val}'),
        hyperparam('--context', 'id', save_dir_key=lambda val: f'{val}'),
        hyperparam('--total-num-context', 5000),
        
        # -- transformer params
        # hyperparam('--over-residual', False, binary_flag=True, save_dir_key=lambda val: f'tres'),
        # hyperparam('--encoder-attention-heads', 1),
        # hyperparam('--encoder-layers', [1], save_dir_key=lambda val: f'enc{val}'),
        # hyperparam('--cross-attention-context', binary_flag=True, save_dir_key=lambda val: f'cac'),
        # hyperparam('--attention-context', binary_flag=True, save_dir_key=lambda val: f'ac'),
        hyperparam('--raymarching-stepsize', 0.05, save_dir_key=lambda val: f'ss{val}'),
        hyperparam('--voxel-size', 0.25, save_dir_key=lambda val: f'v{val}'),
        hyperparam('--max-hits', 60),
        
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
        # hyperparam('--deterministic-step', True, binary_flag=True, save_dir_key=lambda val: 'dstep'),
        hyperparam('--parallel-sampling', True, binary_flag=True, save_dir_key=lambda val: 'ps'),

        # dynamic pruning
        hyperparam('--online-pruning', binary_flag=True, save_dir_key=lambda val: f'pruning'),
        hyperparam('--condition-on-marchsize', binary_flag=True, save_dir_key=lambda val: f'cm'),
        hyperparam('--half-voxel-size-at', "5000,20000,50000", save_dir_key=lambda val: f'dyvox'),
        # hyperparam('--fixed-voxel-size', binary_flag=True, save_dir_key=lambda val: f'fv'),
        hyperparam('--use-max-pruning', binary_flag=True, save_dir_key=lambda val: 'maxp'),

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
        hyperparam('--chunk-size', 256, save_dir_key=lambda val: f'chk512'),
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
        hyperparam('--clip-norm', 0.0),

        hyperparam('--dropout', 0.0),
        hyperparam('--weight-decay', 0.001),
        hyperparam('--criterion', 'srn_loss'),
        hyperparam('--num-workers', 0),
        hyperparam('--seed', [22], save_dir_key=lambda val: f'seed{val}'),
        hyperparam('--save-interval-updates', 500),
        hyperparam('--max-update', 200000),
        
        hyperparam('--virtual-epoch-steps', 5000),
        hyperparam('--save-interval', 1),
        # hyperparam('--no-epoch-checkpoints'),
        hyperparam('--keep-interval-updates', 2),
        hyperparam('--log-format', 'simple'),
        hyperparam('--log-interval', 10 if not args.local else 1),
    ]

    # hyperparams += [
    #     hyperparam('--restore-file', "/checkpoint/jgu/space/neuralrendering/debug_new_chairsv2/srn_data_before2.fp16.seq.128x128.s16.v1.geo_nerf.hyper.emb384.id.ss0.03125.v0.0625.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.fv.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.seed2.ngpu8/checkpoint8.pt")
    # ]
    return hyperparams


@register_grid("geo_shapenet_seq1282_fewshot")
def get_shapenet_seq128fs_grid(args):
    hyperparams = get_shapenet_seq1282_grid(args)
    hyperparams += [
        hyperparam('--train-views', "64", save_dir_key=lambda val: f'fs{val}'),
        # hyperparam('--valid-views', "0..251"),
        hyperparam('--subsample-valid', 1),
        hyperparam('--valid-views', "0,50,100,150,200,250"),
        # hyperparam("--restore-file", 
        #     "/checkpoint/jgu/space/neuralrendering/debug_new_chairs/srn_data_128z.fp16.seq.128x128.s16.v1.geo_nerf_transformer.emb384.id.enc1.cac.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.clip0.0.wd0.0.seed2.ngpu8/checkpoint_best.pt"),
        hyperparam("--freeze-networks", binary_flag=True),
        hyperparam("--reset-context-embed", binary_flag=True),
        hyperparam('--total-num-update', 100000, save_dir_key=lambda val: f'max{val}'),
        hyperparam('--max-update', 100000),
        
        hyperparam('--half-voxel-size-at', "5000,25000", save_dir_key=lambda val: f'dyvox'),
        hyperparam('--save-interval-updates', 500),
        hyperparam('--reset-optimizer', binary_flag=True),
        hyperparam('--reset-meters', binary_flag=True),
        hyperparam('--reset-dataloader', binary_flag=True),
        hyperparam('--reset-lr-scheduler', binary_flag=True),
    ]

    hyperparams += [
        hyperparam('--restore-file',
            "/checkpoint/jgu/space/neuralrendering/debug_new_chairsv3/srn_data_xyz.fp16.seq.128x128.s16.v1.geo_nerf.qxyz.hpc.emb384.id.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.ps.pruning.cm.dyvox.fv.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.seed22.ngpu8/checkpoint_last.pt")
    ]
    return hyperparams


def postprocess_hyperparams(args, config):
    """Postprocess a given hyperparameter configuration."""
    pass


if __name__ == '__main__':
    sweep.main(GRID_REGISTRY, postprocess_hyperparams)
