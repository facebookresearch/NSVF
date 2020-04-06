# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug
DATASET=/private/home/jgu/data/shapenet/${2:-vlad_new}
MODEL_PATH=$1

CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --load-point \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 10 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution 514 \
    --render-combine-output \
    --render-up-vector "(0,1,0)" \
    --render-at-vector "(0,1.5,0)" \
    --render-path-args "{'radius': 2.5, 'h': 1.5, 'axis': 'y', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output3 \
    --render-output-types "rgb" "depth" # "normal" \
#  --render-path-args "{'radius': 3.5, 'h': 0.0, 'axis': 'z'}" \
#     --render-up-vector "(0,0,-1)" \