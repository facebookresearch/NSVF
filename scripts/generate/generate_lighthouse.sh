# just for debugging
DATA=${2:-"lighthouse"}
DATASET=/private/home/jgu/data/shapenet/${DATA}
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution "400x400" \
    --render-path-args "{'radius': 10, 'h': 2, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output6 \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --render-combine-output
    