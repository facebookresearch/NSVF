# just for debugging
DATA=${2:-"hotdog"}
DATASET=/private/home/jgu/data/shapenet/${DATA}
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --load-point \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 10 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution 400 \
    --render-path-args "{'radius': 1.5, 'h': 0.5, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output3 \
    --render-output-types "hit" "rgb" "depth" "normal"  \
    --render-combine-output