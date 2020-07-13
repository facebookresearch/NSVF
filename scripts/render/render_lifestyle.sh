# just for debugging
DATA="lifestyle"
DATASET=/private/home/jgu/data/shapenet/new_renders/data/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/model_${DATA}
ARCH="nsvf_base"
MODEL_PATH=$SAVE/checkpoint_last.pt
# VOXELPLY=$SAVE/edited.ply
# MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"voxel_path":"%s","initial_boundingbox":%s}'
# MODELARGS=$(printf "$MODELTEMP" 1024 0.005 "$VOXELPLY" None)
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.005)

# CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairnr \
#     --task single_object_rendering \
#     --path ${MODEL_PATH} \
#     --render-beam 1 \
#     --render-angular-speed 3 \
#     --render-save-fps 24 \
#     --render-num-frames 20 \
#     --model-overrides $MODELARGS \
#     --render-path-style "circle" \
#     --render-resolution "800x800" \
#     --render-path-args "{'radius': 3, 'h': 1, 'axis': 'z', 't0': -2, 'r':-1}" \
#     --render-output ${SAVE}/output \
#     --render-output-types "color" "depth" "voxel" "normal" \
#     --render-combine-output --log-format "simple" # | tee /checkpoint/jgu/space/neuralrendering/results/${DATA}/rendering.log

CUDA_VISIBLE_DEVICES=0,1,3,4,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-views 200..400 \
    --render-num-frames 20 \
    --model-overrides $MODELARGS \
    --render-camera-poses $DATASET/pose \
    --render-resolution "800x800" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple" # | tee /checkpoint/jgu/space/neuralrendering/results/${DATA}/rendering.log
