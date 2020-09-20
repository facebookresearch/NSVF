# just for debugging
DATA="skull"
# ARCH="nsvf_base"
ARCH="nsvf_xyzn"

DATASET=/private/home/jgu/data/shapenet/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_LingPurple_scaled
NAME=skullxyzn

mkdir -p ${SAVE}_${NAME}

# CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..49" \
    --view-resolution "1200x1600" \
    --max-sentences 1 \
    --view-per-batch 2 \
    --pixel-per-view 4096 \
    --load-mask \
    --no-preload --no-sampling-at-reader \
    --sampling-on-mask 1.0 \
    --valid-view-resolution "600x800" \
    --valid-views "0..48" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch ${ARCH} \
    --voxel-path ${DATASET}/voxel6.txt \
    --raymarching-stepsize 0.0125 \
    --voxel-size 0.1 \
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
    --save-dir ${SAVE}_${NAME} \
    --tensorboard-logdir ${SAVE}/tensorboard/${NAME} \
    | tee -a ${SAVE}_${NAME}/train.log