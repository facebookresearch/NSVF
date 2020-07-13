# just for debugging
DATA=multi_nerf
DATASET=/private/home/jgu/data/shapenet/${DATA}
IDPATH=/private/home/jgu/data/shapenet//multi_nerf/object_ids.txt
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3//geo_nerffullv1.single.800x800.s1.v4.geo_nerf.multi.emb32.addpos6.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu24/checkpoint_best.pt
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_new_full2/geo_nerfv2.single.800x800.s1.v4.geo_nerf.multi.emb32.addpos6.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu24/checkpoint_best.pt
GPU=${3:-0}
FACTOR=8000
GAMMA=0.01

# CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET}/train.txt \
    --user-dir fairnr \
    --task single_object_rendering \
    --model-overrides "{'fp16': False, 'chunk_size': 512, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}, 'object_id_path': '${IDPATH}'}" \
    --path ${MODEL_PATH} \
    --test-views 1 \
    --max-sentences 1 \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 20 \
    --render-resolution "800x1600" \
    --render-path-args "{'radius': 10, 'h': 8, 'axis': 'z', 't0': -1.55, 'r':-1}" \
    --render-output ${MODEL_ROOT}/supplemental/multiobj_circle  \
    --render-output-types "rgb" "hit" --render-combine-output
     
    #"depth" "hit" "normal" \
    