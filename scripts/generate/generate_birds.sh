# just for debugging
ROOT=/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan106/
DATA=scan106
DATASET=${ROOT}/${DATA}
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 24 \
    --render-resolution 600 800 \
    --render-up-vector "(-1,0,0)" \
    --render-path-args "{'radius': 1.5, 'h': 1.0, 'axis': 'x', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output5 \
    --render-output-types "target" "rgb" "hit" "normal" \
    --render-combine-output
