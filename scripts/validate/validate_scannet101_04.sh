DATA="scene0101_04"
DATASET=/private/home/jgu/data/shapenet/scannet/data_render2/${DATA}/newdata
SAVE=/checkpoint/jgu/space/neuralrendering/new_test
SAVE=$SAVE/scannet101_04_bugfix.single.480x640.v1.p2048.w.nsvf_base.vs0.1.ss0.0125.prune2500.adam.lr_poly.max150000.lr0.001.clip0.0.seed20.ngpu32
MODEL_PATH=$SAVE/checkpoint_best.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"tensorboard_logdir":"","eval_lpips":False}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.01)

RES="480x640"
VALID=${1:-"200..1000"}
OUTPUT=${SAVE}/evalw
mkdir -p  ${OUTPUT}

# export CUDA_VISIBLE_DEVICES=0 
python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views ${VALID} \
    --valid-view-resolution ${RES} \
    --no-preload \
    --load-depth \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides $MODELARGS \
    --output-valid ${OUTPUT} \
    # | tee -a ${SAVE}/eval.log