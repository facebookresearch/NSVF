# just for debugging
DATA="scene0101_04"
DATASET=/private/home/jgu/data/shapenet/scannet/data_render2/${DATA}/newdata
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}v3
ARCH="nsvf_base"
BOXZ="bbox.txt"
# ARCH="nsvf_xyz"
mkdir -p $SAVE

CUDA_VISIBLE_DEVICES=0 \
python -u train.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --train-views "0..200" \
    --view-resolution "480x640" \
    --max-sentences 1 \
    --view-per-batch 4 \
    --pixel-per-view 2048 \
    --no-preload \
    --sampling-on-mask 0.85 --sampling-on-bbox \
    --valid-view-resolution "480x640" \
    --valid-views "200..1000:10" \
    --valid-view-per-batch 1 \
    --transparent-background "1.0,1.0,1.0" \
    --background-stop-gradient \
    --arch ${ARCH} \
    --voxel-path ${DATASET}/bbvoxel0.1.txt \
    --voxel-size 0.1 \
    --raymarching-stepsize 0.0125 \
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
    --pruning-every-steps 2500 \
    --keep-interval-updates 5 \
    --log-format simple --log-interval 1 \
    --save-dir ${SAVE} \
    --tensorboard-logdir ${SAVE}/tensorboard \
    | tee -a $SAVE/train.log