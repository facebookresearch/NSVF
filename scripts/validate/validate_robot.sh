DATA="robot"
DATASET=/private/home/jgu/data/shapenet/robot2/${DATA}/scaled
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}w
MODEL_PATH=$SAVE/checkpoint_best.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"tensorboard_logdir":"","eval_lpips":True}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.0)

RES="800x800"
VALID=${1:-"200..400"}
OUTPUT=${SAVE}/eval
mkdir -p  ${OUTPUT}

# export CUDA_VISIBLE_DEVICES=0 
python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views ${VALID} \
    --valid-view-resolution ${RES} \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides $MODELARGS \
    # --output-valid ${OUTPUT} \
    # | tee -a ${SAVE}/eval.log

