# just for debugging
DATA=wineholder
DATASET=/private/home/jgu/data/shapenet/${DATA}/0000
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_wineholder2_latest.single.800x800.s1.v4.geo_nerf.emb32.addpos6.ss0.025.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
MODEL_PATH="geo_wineholder4_jft.single.800x800.s1.v4.geo_nerf.emb32.addpos6.ss0.025.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"

FACTOR=${2:-8000}
GAMMA=${3:-0.01}
GPU=${1:-0}

mkdir -p ${MODEL_ROOT}/supplemental

# python scripts/submission/check_ckpt.py ${MODEL_PATH}

CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_1_2500.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "800x800" \
    --render-camera-poses /checkpoint/jgu/space/neuralrendering/debug_new_singlev3/supplemental/wineholder/pose/0160_800x800.txt \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}_ckpt1_2.5k/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

# python extract.py --path $MODEL_ROOT/$MODEL_PATH/checkpoint_best.pt \
#                   --output ${MODEL_ROOT}/supplemental/${DATA}/ \
#                   --name ${DATA}


# rm -rf ${MODEL_ROOT}/supplemental/${DATA}_zoomin
# python render.py ${DATASET} \
#     --user-dir fairnr \
#     --task single_object_rendering \
#     --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
#     --render-beam 1 \
#     --render-save-fps 24 \
#     --render-resolution "800x800" \
#     --render-path-style "zoomin_line" \
#     --render-num-frames 30 \
#     --render-path-args "{'radius': 3, 'h': 3, 'axis': 'z', 't0': -2, 'r':-1, 'step_r': 240, 'max_r': 2, 'min_r': 0.2}" \
#     --render-output ${MODEL_ROOT}/supplemental/${DATA}_zoomin/ \
#     --render-output-types "rgb" "depth" "hit" "normal" \
#     --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
#     --render-combine-output


# pushd ${MODEL_ROOT}/supplemental/${DATA}
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd

# pushd ${MODEL_ROOT}/supplemental/${DATA}_zoomin
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_zoom.zip rgb/*.png pose/*.txt rgb.mp4
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_zoom.zip NeurIPS2020/SupplementalMaterials/zoomout/
# popd

# pushd ${MODEL_ROOT}/supplemental/${DATA}
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd