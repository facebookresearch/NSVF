# just for debugging
DATA="Truck"
RES="1080x1920"
ARCH="nsvf_base"
SUFFIX="v3"
DATASET=/private/home/jgu/data/shapenet/release/TanksAndTemple/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA
MODEL=$ARCH$SUFFIX
mkdir -p $SAVE/$MODEL

export SLURM_ARGS="""{
    'job-name': '${DATA}-${MODEL}',
    'partition': 'priority',
    'comment': 'NeurIPS open-source',
    'nodes': 1,
    'gpus': 8,
    'output': '$SAVE/$MODEL/train.out',
    'error': '$SAVE/$MODEL/train.stderr.%j',
    'constraint': 'volta32gb',
    'local': True}
"""

python train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..218" \
    --view-resolution $RES \
    --max-sentences 1 \
    --view-per-batch 2 \
    --pixel-per-view 2048 \
    --valid-chunk-size 128 \
    --no-preload\
    --sampling-on-mask 1.0 --no-sampling-at-reader \
    --valid-view-resolution $RES \
    --valid-views "218..250" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch $ARCH \
    --initial-boundingbox ${DATASET}/bbox.txt \
    --raymarching-stepsize-ratio 0.125 \
    --use-octree \
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
    --save-interval-updates 500 --max-update 100000 \
    --virtual-epoch-steps 5000 --save-interval 1 \
    --half-voxel-size-at  "5000,25000" \
    --reduce-step-size-at "5000,25000" \
    --pruning-every-steps 2500 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --tensorboard-logdir ${SAVE}/tensorboard/${MODEL} \
    --save-dir ${SAVE}/${MODEL}
