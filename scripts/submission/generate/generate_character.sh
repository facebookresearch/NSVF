# just for debugging
DATA=blendedmvs_character
DATASET=/private/home/jgu/data/shapenet/final/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_character_finalv1.single.576x768.s1.v4.geo_nerf.emb32.addpos6.ss0.01.v0.08.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
MODEL_PATH="geo_character_finalv1.single.576x768.s1.v4.geo_nerf.emb384.ss0.01.v0.08.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
MODEL_PATH="geo_character_finalnew.single.576x768.s1.v1.geo_nerf.emb32.addpos6.ss0.01.v0.08.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu32"

FACTOR=${2:-8192}
GAMMA=${3:-0.01}
RES=576x768
GPU=${1:-0}
mkdir -p ${MODEL_ROOT}/supplemental

python scripts/submission/check_ckpt.py ${MODEL_PATH}

# CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "576x768" \
    --render-camera-poses ${DATASET}/transform_traj.txt \
    --render-output /private/home/jgu/data/test_images/cutexample \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

#@ --render-output ${MODEL_ROOT}/supplemental/${DATA}_newmodel/ \
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution "576x768" \
#     --render-camera-poses ${DATASET}/zoomout_character \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}_zoomout/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output



# rm -rf ${MODEL_ROOT}/supplemental/${DATA}_zoomin/
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution $RES \
#     --render-path-style "zoomin_line" \
#     --render-num-frames 10 \
#     --render-at-vector "(-0.0683129970398214,0.55959600713104,-0.03020999771025446)" \
#     --render-up-vector "(0,1,0)" \
#     --render-path-args "{'radius': 1, 'h': -1, 'axis': 'y', 't0': -2, 'r':-1, 'step_r': 80, 'max_r': 2, 'min_r': 0.2}" \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}_zoomin/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output

# python extract.py --path $MODEL_ROOT/$MODEL_PATH/checkpoint_best.pt \
#                   --output ${MODEL_ROOT}/supplemental/${DATA}v2/ \
#                   --name ${DATA}

# pushd ${MODEL_ROOT}/supplemental/${DATA}_zoomout
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_zoomout_rgb.zip rgb/*.png rgb.mp4
# # zip -r ${DATA}v2_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_zoomout_rgb.zip NeurIPS2020/SupplementalMaterials/results/nsvf/
# # bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}v2_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd