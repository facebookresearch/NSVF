# just for debugging
DATA=blendedmvs_twopeople
DATASET=/private/home/jgu/data/shapenet/final/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_twopeople_finalv2.fp16.single.576x768.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
MODEL_PATH="geo_twopeople_final_novdir.single.576x768.s1.v4.geo_nerf.emb32.addpos6.ss0.0125.v0.1.posemb.sdfh128.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-8192}
GAMMA=${3:-0.01}
RES=576x768
GPU=${1:-0}
mkdir -p ${MODEL_ROOT}/supplemental

# python scripts/submission/check_ckpt.py ${MODEL_PATH}

# CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution "576x768" \
#     --render-camera-poses ${DATASET}/transform_traj.txt \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}v2/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# python render.py ${DATASET} \
#     --user-dir fairdr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-angular-speed 1.8 \
#     --render-num-frames 25 \
#     --render-resolution ${RES} \
#     --render-up-vector "(0,1,0)" \
#     --render-path-args "{'radius': 1.2, 'h': 0.02, 'axis': 'y', 't0': -2, 'r':-1}" \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}_circle/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output

# python extract.py --path $MODEL_ROOT/$MODEL_PATH/checkpoint_best.pt \
#                   --output ${MODEL_ROOT}/supplemental/${DATA}_circle/ \
#                   --name ${DATA}



CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "576x768" \
    --render-camera-poses ${MODEL_ROOT}/supplemental/_final/${DATA}_circle_pose_adjusted \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}_final_nd/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

# pushd ${MODEL_ROOT}/supplemental/${DATA}_final
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_final_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_final_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_final_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_final_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd