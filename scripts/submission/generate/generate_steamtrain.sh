# just for debugging
DATA=steamtrain
DATASET=/private/home/jgu/data/shapenet/${DATA}/0000
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_steamtrain1_ab2.single.800x800.s1.v4.geo_nerf.emb32.addpos6.ss0.125.v1.0.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-8000}
GAMMA=${3:-0.01}
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
#     --render-resolution "800x800" \
#     --render-camera-poses ${DATASET}/transform_traj.txt \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}/ \
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
#     --render-resolution "800x800" \
#     --render-camera-poses ${MODEL_ROOT}/supplemental/_final/${DATA}_v2 \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}_final/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output


# cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/
# cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/rgb

# pushd ${MODEL_ROOT}/supplemental/${DATA}
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd

pushd ${MODEL_ROOT}/supplemental/${DATA}_final
ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
zip -r ${DATA}_final_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_final_pose.zip pose/*.txt 

bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_final_rgb.zip NeurIPS2020/SupplementalMaterials/results/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_final_pose.zip NeurIPS2020/SupplementalMaterials/pose/
popd