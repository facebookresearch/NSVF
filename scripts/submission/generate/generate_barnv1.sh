# just for debugging
DATA=tanksandtemple_barn
DATASET=/private/home/jgu/data/shapenet/final/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_barn_finalv2.single.1080x1920.s1.v4.geo_nerf.emb384.ss0.1.v0.6.posemb.sdfh256.raydir.r24.bg.bgsg.dis.prune2500.th0.5.dyvox.80k.maxp.m.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-6400}
GAMMA=${3:-0.01}
RES=1080x1920
GPU=${1:-0}
mkdir -p ${MODEL_ROOT}/supplemental

python scripts/submission/check_ckpt.py ${MODEL_PATH}

# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution ${RES} \
#     --render-camera-poses ${DATASET}/transform_traj.txt \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 5 \
    --render-angular-speed 9 \
    --render-num-frames 5 \
    --render-resolution ${RES} \
    --render-at-vector "(-0.26060327957658236,-0.2522170841693879,0.007562308802324134)" \
    --render-up-vector "(0,1,0)" \
    --render-path-args "{'radius': 5.0, 'h': 0.0, 'axis': 'y', 't0': -2, 'r':-1}" \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}_circle/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

# pushd ${MODEL_ROOT}/supplemental/${DATA}_circle
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_circle_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_circle_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_circle_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_circle_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd