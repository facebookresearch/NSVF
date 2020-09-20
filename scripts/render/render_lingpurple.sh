# just for debugging
DATA="LingPurple"
DATASET=/private/home/jgu/data/shapenet/${DATA}/scaled
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}_scaledv2
OUTPUT=$SAVE/output
mkdir -p $OUTPUT

ARCH="nsvf_base"
MODEL_PATH=$SAVE/checkpoint_last.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 512 0.01)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 2 \
    --render-save-fps 4 \
    --render-num-frames 25 \
    --render-at-vector "(0.412,0.945,0.038)" \
    --render-up-vector "(0,-1,0)" \
    --render-path-args "{'radius': 2.5, 'axis': 'y', 'h': 1.5}" \
    --model-overrides $MODELARGS \
    --render-resolution "940x1285" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "normal" \
    --render-combine-output --log-format "simple"

