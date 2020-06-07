MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
DATA=/private/home/jgu/data/shapenet//multi_nerf/train_full.txt
IDPATH=/private/home/jgu/data/shapenet//multi_nerf/object_ids.txt
MODEL=geo_nerffullv1.single.800x800.s1.v4.geo_nerf.multi.emb32.addpos6.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu24
RES="800x800"
VALID="200..400"
FACTOR=8000
GAMMA=0.01
OUTPUT=$MODEL_ROOT/test_output/${MODEL}

mkdir -p  ${OUTPUT}/${GAMMA}

# export CUDA_VISIBLE_DEVICES=0
python validate.py \
    $DATA \
    --valid-views ${VALID} \
    --valid-view-resolution ${RES} \
    --user-dir fairdr \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_ROOT}/${MODEL}/checkpoint_best.pt \
    --model-overrides "{'chunk_size': 512, 'aabb_factor': ${FACTOR}, 'use_lpips': True, 'parallel_sampling': False, 'raymarching_tolerance': ${GAMMA}, 'object_id_path': '${IDPATH}'}" \
    --output-valid ${OUTPUT}/${GAMMA} \

# bash scripts/validate/validate_nerf_test.sh   800x800 200..400 8000 0.01 