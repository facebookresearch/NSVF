# just for debugging
ROOT=/private/home/jgu/data/shapenet/
DATA=ignatius_srn
# WORK=/checkpoint/jgu/space/neuralrendering/debug_ignatius
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_single
DATASET=${ROOT}/${DATA}

MODEL1=geo_ignatiusp.fp16.single.540x960.s1.v2.geo_nerf.emb384.ss0.1.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL2=geo_ignatius22p.single.540x960.s1.v1.geo_nerf.emb384.ss0.08.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL3=geo_ignatius.fp16.single.540x960.s1.v16.geo_nerf.emb384.ss0.1.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL4=geo_ignatius_reload_check.fp16.single.540x960.s1.v16.geo_nerf.emb384.ss0.1.v0.5.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.m.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL5=geo_ignatiusnewv201.fp16.single.540x960.s1.v4.geo_nerf.emb384.ss0.04.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.maxp.m.p4096.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL5=geo_ignatius2sp201.fp16.single.540x960.s1.v4.geo_nerf.emb384.ss0.04.v0.2.posemb.sdfh128.raydir.r24.spec.sd0.5.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.80k.maxp.m.p4096.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8

CHECKPOINT=checkpoint_last.pt
# CHECKPOINT=checkpoint1.pt
# CHECKPOINT=checkpoint_1_2500.pt

MODEL_PATH=${WORK}/${MODEL5}/${CHECKPOINT}
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution "540x960" \
    --render-up-vector "(0,1,0)" \
    --render-path-args "{'radius': 3.5, 'h': 0.0, 'axis': 'y', 't0': -2, 'r':-1}" \
    --render-output /checkpoint/jgu/space/neuralrendering/results/${DATA}/ \
    --render-output-types "hit" "rgb" "normal" \
    --render-combine-output
