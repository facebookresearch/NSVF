# just for debugging
DATA="Jade"
RES="576x768"
ARCH="nsvf_emb0"
DATASET=/private/home/jgu/data/shapenet/release/BlendedMVS/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA
mkdir -p $SAVE/$ARCH

SLURM_ARGS="""
{   'job-name': '${DATA}-${ARCH}',
    'partition': 'priority',
    'comment': 'NeurIPS open-source',
    'nodes': 1,
    'gpus': 8,
    'output': '$SAVE/$ARCH/train.out',
    'error': '$SAVE/$ARCH/train.stderr.%j',
    'constraint': 'volta32gb',
    'local': False}
"""

python train.py ${DATASET} \
    --slurm-args ${SLURM_ARGS//[[:space:]]/} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..50" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "50..58" \
    --valid-view-per-batch 1 \
    --transparent-background "0.0,0.0,0.0" \
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
