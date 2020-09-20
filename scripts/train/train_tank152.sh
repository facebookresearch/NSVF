# just for debugging
DATA="isu152"
DATASET=/private/home/jgu/data/shapenet/new_renders/data/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}w
ARCH="nsvf_base"
BOXZ="bbox.txt"
# ARCH="nsvf_xyz"
mkdir -p $SAVE

# CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..100" \
    --view-resolution "800x800" \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 0.8 --sampling-on-bbox \
    --valid-view-resolution "400x400" \
    --valid-views "100..196" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch ${ARCH} \
    --initial-boundingbox ${DATASET}/${BOXZ} \
    --raymarching-stepsize 0.02 \
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
    --save-dir ${SAVE} \
    --tensorboard-logdir ${SAVE}/tensorboard \
    | tee -a $SAVE/train.log