# just for debugging
DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/fffda9f09223a21118ff2740a556cc3
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug16
ARCH=dvr_base
# FAIRSEQ=/private/home/jgu/work/fairseq-master/fairseq_cli

mkdir -p $MODEL_PATH

CUDA_VISIBLE_DEVICES=0,1 \
fairseq-train $DATASET \
    --user-dir fairdr/ \
    --save-dir $MODEL_PATH \
    --tensorboard-logdir $MODEL_PATH/tensorboard \
    --max-sentences 1 \
    --pixel-per-view 2048 \
    --view-per-batch 25 \
    --task single_object_rendering \
    --criterion dvr_loss \
    --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --L1 \
    --lr 0.0001 --lr-scheduler fixed \
    --save-interval-updates 1000 \
    --validate-interval 100 \
    --save-interval 100000 \
    --max-update 300000 \
    --no-epoch-checkpoints \
    --log-format simple \
    --log-interval 10 \
    
    # --rgb-weight 1.0 --space-weight 100 --occupancy-weight 100
    # --min-lr '1e-09' --warmup-updates 1000 \
    # --warmup-init-lr '1e-07' \