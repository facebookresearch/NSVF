# just for debugging
DATA="wineholder"
RES="800x800"
ARCH="nsvf_xyz"
DATASET=/private/home/jgu/data/shapenet/${DATA}/scaled2
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/test_$DATA

mkdir -p $SAVE

SLURM_ARGS="""
{   'job-name': 'nsvf',
    'partition': 'priority',
    'comment': 'ICLR2021',
    'nodes': 1,
    'gpus': 8,
    'output': '$SAVE/$ARCH/train.out',
    'error': '$SAVE/$ARCH/train.%j.err',
    'constraint': 'volta32gb',
    'local': True}
"""

# CUDA_VISIBLE_DEVICES=0 \
python train.py ${DATASET} \
    --slurm-args ${SLURM_ARGS//[[:space:]]/} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..100" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "100..200" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --discrete-regularization \
    --color-weight 128.0 \
    --alpha-weight 1.0 \
    --optimizer "adam" \
    --adam-betas "(0.9, 0.999)" \
    --lr-scheduler "polynomial_decay" \
    --total-num-update 150000 \
    --lr 0.001 \
    --clip-norm 0.0 \
    --criterion "srn_loss" \
    --num-workers 0 \
    --seed 2 \
    --save-interval-updates 500 --max-update 150000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "5000,25000,75000" \
    --reduce-step-size-at "5000,25000,75000" \
    --pruning-every-steps 2500 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${ARCH} \
    --save-dir ${SAVE}/${ARCH}