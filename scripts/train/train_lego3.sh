# just for debugging
DATA="lego_full"
# ARCH="nsvf_base"
# ARCH="nsvf_xyz"
ARCH="nsvf_embn"

DATASET=/private/home/jgu/data/shapenet/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_LingPurple_scaled

mkdir -p ${SAVE}_lego0embn

# CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..100" \
    --view-resolution "800x800" \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --no-preload --no-sampling-at-reader \
    --sampling-on-mask 1.0 \
    --valid-view-resolution "800x800" \
    --valid-views "100..196" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch ${ARCH} \
    --voxel-path ${DATASET}/voxel.txt \
    --raymarching-stepsize 0.05 \
    --voxel-size 0.4 \
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
    --save-dir ${SAVE}_lego0embn \
    --tensorboard-logdir ${SAVE}/tensorboard/lego0embn \
    | tee -a ${SAVE}_lego0embn/train.log