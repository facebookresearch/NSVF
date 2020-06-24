# just for debugging
DATA=blendedmvs_jade
DATASET=/private/home/jgu/data/shapenet/final/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_jade_finalv5.single.576x768.s1.v4.geo_nerf.emb384.ss0.00375.v0.03.posemb.sdfh128.raydir.r24.bg0.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-8192}
GAMMA=${3:-0.01}
GPU=${1:-0}
mkdir -p ${MODEL_ROOT}/supplemental

python scripts/submission/check_ckpt.py ${MODEL_PATH}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "576x768" \
    --render-camera-poses ${MODEL_ROOT}/supplemental/_final/${DATA}v2_pose_adjusted \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}_final/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

# # cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/
# # cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/rgb

# python extract.py --path $MODEL_ROOT/$MODEL_PATH/checkpoint_best.pt \
#                   --output ${MODEL_ROOT}/supplemental/${DATA}v2/ \
#                   --name ${DATA}

# pushd ${MODEL_ROOT}/supplemental/${DATA}v2
# ffmpeg -framerate 20 -pattern_type glob -i 'rgb/*.png' rgb.mp4
# zip -r ${DATA}v2_rgb.zip rgb/*.png rgb.mp4
# zip -r ${DATA}v2_pose.zip pose/*.txt 

# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}v2_rgb.zip NeurIPS2020/SupplementalMaterials/nsvf/
# bash ~/tools/Dropbox-Uploader/dropbox_uploader.sh upload ${DATA}v2_pose.zip NeurIPS2020/SupplementalMaterials/pose/
# popd