# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
DATASET=/private/home/jgu/data/shapenet/maria
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn10
ARCH=srn_base
CRITERION=srn_loss
# FAIRSEQ=/private/home/jgu/work/fairseq-master/fairseq_cli

mkdir -p $MODEL_PATH

# CUDA_VISIBLE_DEVICES=0,1 \
fairseq-train $DATASET \
    --user-dir fairnr/ \
    --save-dir $MODEL_PATH \
    --tensorboard-logdir $MODEL_PATH/tensorboard \
    --load-depth \
    --max-sentences 1 \
    --view-per-batch 5 \
    --view-resolution 128 \
    --raymarching-steps 8 \
    --rgb-weight 200 --reg-weight 1e-3 --depth-weight 0.01 \
    --task single_object_rendering \
    --criterion $CRITERION \
    --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.999)' \
    --clip-norm 0.0 \
    --lr 0.002 --lr-scheduler fixed \
    --save-interval-updates 1000 \
    --validate-interval 20 \
    --save-interval 100000 \
    --max-update 10000 \
    --no-epoch-checkpoints \
    --log-format simple \
    --log-interval 20 \
    
    # --rgb-weight 1.0 --space-weight 100 --occupancy-weight 100
    # --min-lr '1e-09' --warmup-updates 1000 \
    # --warmup-init-lr '1e-07' \