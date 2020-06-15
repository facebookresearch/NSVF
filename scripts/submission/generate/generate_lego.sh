# just for debugging
DATA=lego_full
DATASET=/private/home/jgu/data/shapenet/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_legov2.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-8000}
GAMMA=${3:-0.01}
GPU=${1:-0}

mkdir -p ${MODEL_ROOT}/supplemental

# python scripts/submission/check_ckpt.py ${MODEL_PATH}
# python scripts/submission/get_test_pose.py ${DATASET}

# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution "800x800" \
#     --render-camera-poses ${DATASET}/test_traj.txt \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output

rm -rf ${MODEL_ROOT}/supplemental/${DATA}_zoomin
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "800x800" \
    --render-path-style "zoomin_line" \
    --render-num-frames 30 \
    --render-path-args "{'radius': 1, 'h': 2, 'axis': 'z', 't0': 0, 'r':-1, 'step_r': 240, 'max_r': 3, 'min_r': 0.2}" \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}_zoomin/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output


pushd ${MODEL_ROOT}/supplemental/${DATA}_zoomin
ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
zip -r ${DATA}_zoom.zip rgb/*.png pose/*.txt rgb.mp4
bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_zoom.zip NeurIPS2020/SupplementalMaterials/zoomout/
popd