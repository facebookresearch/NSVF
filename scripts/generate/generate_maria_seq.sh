# just for debugging
DATA=${2:-"maria_seq"}
DATASET=/private/home/jgu/data/shapenet/maria/${DATA}.txt
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --test-views 25 \
    --load-point \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 1 \
    --max-sentences 10 \
    --render-angular-speed 0 \
    --render-save-fps 24 \
    --render-num-frames 1 \
    --render-resolution 512 \
    --render-path-args "{'radius': 3.0, 'h': 0.5, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output4 \
    --render-output-types "target" "rgb" "hit"\
    --render-combine-output
    # "hit" "normal" "depth"
    