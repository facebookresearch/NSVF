# just for debugging
DATA=${2:-"ficus"}
DATASET=/private/home/jgu/data/shapenet/${DATA}
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --load-point \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution "400x400" \
    --render-path-args "{'radius': 4, 'h': 0, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /checkpoint/jgu/space/neuralrendering/results/${DATA}/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --render-combine-output
    