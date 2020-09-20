# just for debugging
DATA="LingPurple"
DATASET=/private/home/jgu/data/shapenet/${DATA}/scaled
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}_scaled
ARCH="nsvf_base"
BOXZ="bbox.txt"
# ARCH="nsvf_xyz"
mkdir -p $SAVE

# --trainable-extrinsics \
# CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..101" \
    --view-resolution "940x1285" \
    --max-sentences 1 \
    --view-per-batch 2 \
    --pixel-per-view 2048 \
    --trainable-extrinsics \
    --no-sampling-at-reader \
    --no-preload \
    --sampling-on-mask 1.0 \
    --valid-view-resolution "940x1285" \
    --valid-views "0..48" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --arch ${ARCH} \
    --voxel-path ${DATASET}/voxel_0.08.txt \
    --voxel-size 0.08 \
    --raymarching-stepsize 0.01 \
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
    --save-dir ${SAVE}v3v2 \
    --tensorboard-logdir ${SAVE}/tensorboard/v3v2 \
    | tee -a $SAVE/train.log