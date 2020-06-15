# just for debugging
DATA="wineholder"
DATASET=/private/home/jgu/data/shapenet/${DATA}/0000
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL=geo_wineholder2_latest.single.800x800.s1.v4.geo_nerf.emb32.addpos6.ss0.025.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL_PATH=$MODEL_ROOT/$MODEL/checkpoint_last.pt
GPU=${3:-0}

# CUDA_VISIBLE_DEVICES=0,1 \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 125 \
    --render-path-style "zoomin_line" \
    --render-resolution "400x400" \
    --render-path-args "{'radius': 4, 'h': 4, 'axis': 'z', 't0': -2, 'r':-1, 'step_r': 1000, 'max_r': 10, 'min_r': 0.00001}" \
    --render-output /checkpoint/jgu/space/neuralrendering/results/${DATA}/ \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --render-combine-output | tee /checkpoint/jgu/space/neuralrendering/results/${DATA}/rendering.log