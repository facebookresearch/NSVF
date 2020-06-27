# just for debugging
DATA="wineholder"
DATASET=/private/home/jgu/data/shapenet/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_codebase/model2
MODEL_PATH=$SAVE/checkpoint_last.pt

# CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 15 \
    --model-overrides "{'chunk_size': 1024, 'raymarching_tolerance': 0.01}" \
    --render-path-style "circle" \
    --render-resolution "800x800" \
    --render-path-args "{'radius': 3, 'h': 2, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple" # | tee /checkpoint/jgu/space/neuralrendering/results/${DATA}/rendering.log
