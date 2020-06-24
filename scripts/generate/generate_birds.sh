# just for debugging
ROOT=/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan106/
DATA=scan106
DATASET=${ROOT}/${DATA}
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_birds
MODEL1=scan106_v3.fp16.single.600x800.s1.v2.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.m.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8

MODEL_PATH=${WORK}/${MODEL1}
GPU=${1:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --model-overrides "{'total_num_embedding': 12000}" \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution "600x800" \
    --render-up-vector "(0,0,1)" \
    --render-path-args "{'radius': 2.5, 'h': -1.5, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /checkpoint/jgu/space/neuralrendering/results/${DATA}/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --render-combine-output
