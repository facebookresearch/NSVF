# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os, copy
import json
import torch
import imageio
import numpy as np
from collections import defaultdict
from torchvision.utils import save_image
from argparse import Namespace

from fairseq.tasks import FairseqTask, register_task
from fairseq.optim.fp16_optimizer import FP16Optimizer
from fairseq.logging import progress_bar

from fairnr.data import (
    ShapeViewDataset, SampledPixelDataset, ShapeViewStreamDataset,
    WorldCoordDataset, ShapeDataset, InfiniteDataset
)
from fairnr.data.data_utils import write_images, recover_image, parse_views
from fairnr.data.geometry import ray, compute_normal_map
from fairnr.renderer import NeuralRenderer
from fairnr.data.trajectory import get_trajectory
from fairnr import ResetTrainerException


@register_task("single_object_rendering")
class SingleObjRenderingTask(FairseqTask):
    """
    Task for remembering & rendering a single object.
    """
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser"""
        parser.add_argument("data", help='data-path or data-directoy')
        parser.add_argument("--object-id-path", type=str, help='path to object indices', default=None)
        parser.add_argument("--no-preload", action="store_true")
        parser.add_argument("--no-load-binary", action="store_true")
        parser.add_argument("--load-depth", action="store_true", 
                            help="load depth images if exists")
        parser.add_argument("--transparent-background", type=str, default="1.0",
                            help="background color if the image is transparent")
        parser.add_argument("--load-mask", action="store_true",
                            help="load pre-computed masks which is useful for subsampling during training.")
        parser.add_argument("--train-views", type=str, default="0..50", 
                            help="views sampled for training, you can set specific view id, or a range")
        parser.add_argument("--valid-views", type=str, default="0..50",
                            help="views sampled for validation,  you can set specific view id, or a range")
        parser.add_argument("--test-views", type=str, default="0",
                            help="views sampled for rendering, only used for showing rendering results.")
        parser.add_argument("--subsample-valid", type=int, default=-1,
                            help="if set > -1, subsample the validation (when training set is too large)")
        parser.add_argument("--view-per-batch", type=int, default=6,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--valid-view-per-batch", type=int, default=1,
                            help="number of views training each batch (each GPU)")
        parser.add_argument("--view-resolution", type=str, default='64x64',
                            help="width for the squared image. downsampled from the original.")    
        parser.add_argument('--valid-view-resolution', type=str, default=None,
                            help="if not set, if valid view resolution will be train view resolution")   
        parser.add_argument("--min-color", choices=(0, -1), default=-1, type=int,
                            help="RGB range used in the model. conventionally used -1 ~ 1")
        parser.add_argument("--virtual-epoch-steps", type=int, default=None,
                            help="virtual epoch used in Infinite Dataset. if None, set max-update")
        parser.add_argument("--pruning-every-steps", type=int, default=None,
                            help="if the model supports pruning, prune unecessary voxels")
        parser.add_argument("--half-voxel-size-at", type=str, default=None,
                            help='specific detailed number of updates to half the voxel sizes')
        parser.add_argument("--reduce-step-size-at", type=str, default=None,
                            help='specific detailed number of updates to reduce the raymarching step sizes')
        parser.add_argument("--prune-voxel-at", type=str, default=None,
                            help='specific detailed number of pruning voxels')
        parser.add_argument("--rendering-every-steps", type=int, default=None,
                            help="if set, enables rendering online with default parameters")
        parser.add_argument("--rendering-args", type=str, metavar='JSON')
        parser.add_argument("--pruning-th", type=float, default=0.5,
                            help="if larger than this, we choose keep the voxel.")
        parser.add_argument("--pruning-with-train-stats", action='store_true',
                            help="if set, model will run over the training set statstics to prune voxels.")
        parser.add_argument("--pruning-rerun-train-set", action='store_true',
                            help="only works when --pruning-with-train-stats is also set.")
        parser.add_argument("--output-valid", type=str, default=None)

    def __init__(self, args):
        super().__init__(args)
        
        self._trainer, self._dummy_batch = None, None

        # check dataset
        self.train_data = self.val_data = self.test_data = args.data
        self.object_ids = None if args.object_id_path is None else \
            {line.strip(): i for i, line in enumerate(open(args.object_id_path))}
        self.output_valid = getattr(args, "output_valid", None)
        
        if os.path.isdir(args.data):
            if os.path.exists(args.data + '/train.txt'):
                self.train_data = args.data + '/train.txt'
            if os.path.exists(args.data + '/val.txt'):
                self.val_data = args.data + '/val.txt'
            if os.path.exists(args.data + '/test.txt'):
                self.test_data = args.data + '/test.txt'
            if self.object_ids is None and os.path.exists(args.data + '/object_ids.txt'):
                self.object_ids = {line.strip(): i for i, line in enumerate(open(args.data + '/object_ids.txt'))}
        if self.object_ids is not None:
            self.ids_object = {self.object_ids[o]: o for o in self.object_ids}
        else:
            self.ids_object = {0: 'model'}

        if len(self.args.tensorboard_logdir) > 0 and getattr(args, "distributed_rank", -1) == 0:
            from tensorboardX import SummaryWriter
            self.writer = SummaryWriter(self.args.tensorboard_logdir + '/images')
        else:
            self.writer = None

        self._num_updates = {'pv': -1, 'sv': -1, 'rs': -1, 're': -1}
        self.pruning_every_steps = getattr(self.args, "pruning_every_steps", None)
        self.pruning_th = getattr(self.args, "pruning_th", 0.5)
        self.rendering_every_steps = getattr(self.args, "rendering_every_steps", None)
        self.steps_to_half_voxels = getattr(self.args, "half_voxel_size_at", None)
        self.steps_to_reduce_step = getattr(self.args, "reduce_step_size_at", None)
        self.steps_to_prune_voxels = getattr(self.args, "prune_voxel_at", None)

        if self.steps_to_half_voxels is not None:
            self.steps_to_half_voxels = [int(s) for s in self.steps_to_half_voxels.split(',')]
        if self.steps_to_reduce_step is not None:
            self.steps_to_reduce_step = [int(s) for s in self.steps_to_reduce_step.split(',')]
        if self.steps_to_prune_voxels is not None:
            self.steps_to_prune_voxels = [int(s) for s in self.steps_to_prune_voxels.split(',')]

        if self.rendering_every_steps is not None:
            gen_args = {
                'path': args.save_dir,
                'render_beam': 1, 'render_resolution': '512x512',
                'render_num_frames': 120, 'render_angular_speed': 3,
                'render_output_types': ["rgb"], 'render_raymarching_steps': 10,
                'render_at_vector': "(0,0,0)", 'render_up_vector': "(0,0,-1)",
                'render_path_args': "{'radius': 1.5, 'h': 0.5}",
                'render_path_style': 'circle', "render_output": None
            }
            gen_args.update(json.loads(getattr(args, 'rendering_args', '{}') or '{}'))
            self.renderer = self.build_generator(Namespace(**gen_args))    
        else:
            self.renderer = None

        self.train_views = parse_views(args.train_views)
        self.valid_views = parse_views(args.valid_views)
        self.test_views  = parse_views(args.test_views)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """
        Setup the task
        """
        return cls(args)

    def repeat_dataset(self, split):
        return 1 if split != 'train' else self.args.distributed_world_size  # IMPORTANT!

    def load_dataset(self, split, **kwargs):
        """
        Load a given dataset split (train, valid, test)
        """
        self.datasets[split] = ShapeViewDataset(
            self.train_data if split == 'train' else \
                self.val_data if split == 'valid' else self.test_data,
            views=self.train_views if split == 'train' else \
                self.valid_views if split == 'valid' else self.test_views,
            num_view=self.args.view_per_batch if split == 'train' else \
                self.args.valid_view_per_batch if split == 'valid' else 1,
            resolution=self.args.view_resolution if split == 'train' else \
                getattr(self.args, "valid_view_resolution", self.args.view_resolution) if split == 'valid' else \
                    getattr(self.args, "render_resolution", self.args.view_resolution),
            subsample_valid=self.args.subsample_valid if split == 'valid' else -1,
            train=(split=='train'),
            load_depth=self.args.load_depth and (split!='test'),
            load_mask=self.args.load_mask and (split!='test'),
            repeat=self.repeat_dataset(split),
            preload=(not getattr(self.args, "no_preload", False)) and (split!='test'),
            binarize=(not getattr(self.args, "no_load_binary", False)) and (split!='test'),
            bg_color=getattr(self.args, "transparent_background", "1,1,1"),
            min_color=getattr(self.args, "min_color", -1),
            ids=self.object_ids
        )

        if split == 'train':
            max_step = getattr(self.args, "virtual_epoch_steps", None)
            if max_step is not None:
                total_num_models = max_step * self.args.distributed_world_size * self.args.max_sentences
            else:
                total_num_models = 10000000

            if getattr(self.args, "pruning_rerun_train_set", False):
                self._unique_trainset = ShapeViewStreamDataset(copy.deepcopy(self.datasets[split]))  # backup
                self._unique_trainitr = self.get_batch_iterator(
                    self._unique_trainset, max_sentences=self.args.max_sentences_valid, seed=self.args.seed,
                    num_shards=self.args.distributed_world_size, shard_id=self.args.distributed_rank, 
                    num_workers=self.args.num_workers)
            self.datasets[split] = InfiniteDataset(self.datasets[split], total_num_models)

        if split == 'valid':
            self.datasets[split] = ShapeViewStreamDataset(self.datasets[split])

    def build_generator(self, args):
        """
        build a neural renderer for visualization
        """
        return NeuralRenderer(
            beam=args.render_beam,
            resolution=args.render_resolution,
            frames=args.render_num_frames,
            speed=args.render_angular_speed,
            raymarching_steps=args.render_raymarching_steps,
            path_gen=get_trajectory(args.render_path_style)(
                **eval(args.render_path_args)
            ),
            at=eval(args.render_at_vector),
            up=eval(args.render_up_vector),
            fps=getattr(args, "render_save_fps", 24),
            output_dir=args.render_output if args.render_output is not None
                else os.path.join(args.path, "output"),
            output_type=args.render_output_types,
            test_camera_poses=getattr(args, "render_camera_poses", None),
            test_camera_intrinsics=getattr(args, "render_camera_intrinsics", None),
            test_camera_views=getattr(args, "render_views", None)
        )

    def setup_trainer(self, trainer):
        # give the task ability to access the global trainer functions
        self._trainer = trainer

    @property
    def source_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None
    
    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return None

    def update_step(self, num_updates, name='re'):
        """Task level update when number of updates increases.

        This is called after the optimization step and learning rate
        update at each iteration.
        """
        self._num_updates[name] = num_updates

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        if (((self.pruning_every_steps is not None) and \
            (update_num % self.pruning_every_steps == 0) and \
            (update_num > 0)) or \
            ((self.steps_to_prune_voxels is not None) and \
             update_num in self.steps_to_prune_voxels) \
             ) and \
            (update_num > self._num_updates['pv']) and \
            hasattr(model, 'prune_voxels'):
            model.eval()
            if getattr(self.args, "pruning_rerun_train_set", False):
                with torch.no_grad():
                    model.clean_caches(reset=True)
                    progress = progress_bar.progress_bar(
                        self._unique_trainitr.next_epoch_itr(shuffle=False),
                        prefix=f"pruning based statiscs over training set",
                        tensorboard_logdir=None, 
                        default_log_format=self.args.log_format if self.args.log_format is not None else "tqdm")
                    for step, inner_sample in enumerate(progress):
                        outs = model(**self._trainer._prepare_sample(self.filter_dummy(inner_sample)))
                        progress.log(stats=outs['other_logs'], tag='track', step=step)

            model.prune_voxels(self.pruning_th, train_stats=getattr(self.args, "pruning_with_train_stats", False))
            self.update_step(update_num, 'pv')

        if self.steps_to_half_voxels is not None and \
            (update_num in self.steps_to_half_voxels) and \
            (update_num > self._num_updates['sv']):
            
            model.split_voxels()
            self.update_step(update_num, 'sv')
            raise ResetTrainerException

        if self.rendering_every_steps is not None and \
            (update_num % self.rendering_every_steps == 0) and \
            (update_num > 0) and \
            self.renderer is not None and \
            (update_num > self._num_updates['re']):

            sample_clone = {key: sample[key].clone() if sample[key] is not None else None for key in sample }
            outputs = self.inference_step(self.renderer, [model], [sample_clone, 0])[1]
            if getattr(self.args, "distributed_rank", -1) == 0:  # save only for master
                self.renderer.save_images(outputs, update_num)
            self.steps_to_half_voxels = [a for a in self.steps_to_half_voxels if a != update_num]

        if self.steps_to_reduce_step is not None and \
            update_num in self.steps_to_reduce_step and \
            (update_num > self._num_updates['rs']):

            model.reduce_stepsize()
            self.update_step(update_num, 'rs')
        
        self.update_step(update_num, 'step')
        return super().train_step(sample, model, criterion, optimizer, update_num, ignore_grad)

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        model.add_eval_scores(logging_output, sample, model.cache, criterion, outdir=self.output_valid)
        if self.writer is not None:
            images = model.visualize(sample, shape=0, view=0)
            if images is not None:
                write_images(self.writer, images, self._num_updates['step'])
        
        return loss, sample_size, logging_output
    
    def save_image(self, img, id, view, group='gt'):
        object_name = self.ids_object[id.item()]
        def _mkdir(x):
            if not os.path.exists(x):
                os.mkdir(x)
        _mkdir(self.output_valid)
        _mkdir(os.path.join(self.output_valid, group))  
        _mkdir(os.path.join(self.output_valid, group, object_name))
        imageio.imsave(os.path.join(
            self.output_valid, group, object_name, 
            '{:04d}.png'.format(view)), 
            (img * 255).astype(np.uint8))

    def filter_dummy(self, sample):
        if self._dummy_batch is None:
            self._dummy_batch = sample
        if sample is None:
            sample = self._dummy_batch
        return sample
