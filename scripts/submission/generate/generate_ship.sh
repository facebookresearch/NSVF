# just for debugging
DATA=ship_full
DATASET=/private/home/jgu/data/shapenet/${DATA}
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH="geo_shipv1.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8"
FACTOR=${2:-6400}
GAMMA=${3:-0.01}
GPU=${1:-0}

mkdir -p ${MODEL_ROOT}/supplemental

# 
# CUDA_VISIBLE_DEVICES=0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --load-point \
    --path ${MODEL_ROOT}/${MODEL_PATH}/checkpoint_best.pt \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-resolution "800x800" \
    --render-camera-poses ${DATASET}/test_traj.txt \
    --render-output ${MODEL_ROOT}/supplemental/${DATA}/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-combine-output

# cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/
# cp ${DATASET}/test_traj.txt ${MODEL_ROOT}/supplemental/${DATA}/rgb

pushd ${MODEL_ROOT}/supplemental/${DATA}
zip -r output_rgb.zip rgb/*.png pose/*.png 
popd