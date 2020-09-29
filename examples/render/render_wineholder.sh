# just for debugging
DATA="wineholder"
RES="800x800"
ARCH="nsvf_base"
DATASET=/private/home/jgu/data/shapenet/${DATA}/scaled2
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/test_$DATA
MODEL_PATH=$SAVE/checkpoint_last.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.01)

# CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "200..400" \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"