# just for debugging
DATA="ficus_full"
ARCH='nsvf_base'
DATASET=/private/home/jgu/data/shapenet/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_LingPurple_scaled
TESTPOSE=$DATASET/test_traj.txt
NAME="ficus"_${ARCH}v2

MODEL_PATH=${SAVE}_${NAME}/checkpoint_last.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.01)

# CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 20 \
    --render-camera-poses $TESTPOSE \
    --model-overrides $MODELARGS \
    --render-resolution "800x800" \
    --render-output ${SAVE}_$NAME/test_output \
    --render-output-types "color" "normal" \
    --render-combine-output --log-format "simple"
