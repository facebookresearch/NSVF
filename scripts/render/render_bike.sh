# just for debugging
DATA="bike"
DATASET=/private/home/jgu/data/shapenet/new_renders/data/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}
TESTPOSE=/private/home/jgu/data/shapenet/new_renders/data/pose_test/pose_2.5m/
OUTPUT=/private/home/jgu/data/shapenet/new_renders/data/pose_test/nsvf_results

mkdir -p $OUTPUT

ARCH="nsvf_base"
MODEL_PATH=$SAVE/checkpoint_last.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.01)

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
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
    --render-output ${SAVE}/test_output25m \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"

cp -r ${SAVE}/test_output25m/color $OUTPUT/${DATA}25m