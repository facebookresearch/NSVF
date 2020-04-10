# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
DATASET=/private/home/jgu/data/shapenet/maria
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn_ref25
SLURM_MODEL=/checkpoint/jgu/space/neuralrendering/slurm_srn/maria.no_c10d.single.512x512.s1.v5.p16384.mask0.75.march10.rgb200.dep0.08.vgg1.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8/
ARCH=srn_base
CRITERION=srn_loss
# FAIRSEQ=/private/home/jgu/work/fairseq-master/fairseq_cli
# --restore-file $SLURM_MODEL/checkpoint_last.pt \
mkdir -p $MODEL_PATH

# CUDA_VISIBLE_DEVICES=0 \
python train.py $DATASET \
    --ddp-backend no_c10d \
    --user-dir fairdr/ \
    --save-dir $MODEL_PATH \
    --tensorboard-logdir $MODEL_PATH/tensorboard \
    --max-sentences 1 \
    --pixel-per-view 16384 \
    --sampling-on-mask 0.75 \
    --sampling-on-bbox \
    --view-per-batch 5 \
    --view-resolution 512 \
    --raymarching-steps 10 \
    --load-depth --load-mask \
    --rgb-weight 200 --reg-weight 1e-3 --depth-weight 0.08 --vgg-weight 1.0 \
    --task single_object_rendering \
    --criterion $CRITERION \
    --arch $ARCH \
    --optimizer adam --adam-betas '(0.9, 0.999)' \
    --clip-norm 0.0 \
    --lr 0.001 --lr-scheduler fixed \
    --save-interval-updates 500 \
    --validate-interval 100 \
    --save-interval 100000 \
    --max-update 300000 \
    --no-epoch-checkpoints \
    --log-format simple \
    --log-interval 20 \
    
    # --fp16
    # --load-depth
    # /datasets01/scannet/082518/scans/scene0172_00
    # --rgb-weight 1.0 --space-weight 100 --occupancy-weight 100
    # --min-lr '1e-09' --warmup-updates 1000 \
    # --warmup-init-lr '1e-07' \