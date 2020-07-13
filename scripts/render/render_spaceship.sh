# just for debugging
DATA="spaceship2"
DATASET=/private/home/jgu/data/shapenet/new_renders/data/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}
ARCH="nsvf_base"
MODEL_PATH=$SAVE/checkpoint_last.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.005)

CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 20 \
    --render-camera-poses $DATASET/pose \
    --render-views 200..400 \
    --model-overrides $MODELARGS \
    --render-path-style "circle" \
    --render-resolution "800x800" \
    --render-path-args "{'radius': 2, 'h': 1.4, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"