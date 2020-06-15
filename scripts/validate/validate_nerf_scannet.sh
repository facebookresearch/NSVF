MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_scannetv2
DATA=/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0101_04/data
MODEL=scene0101_04vb.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL=scene0101_04nodepth.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent0.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8

RES="480x640"
VALID="0..1000"
FACTOR=8000
GAMMA=0.01
OUTPUT=$MODEL_ROOT/test_output2/${MODEL}

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
    --model-overrides "{'chunk_size': 512, 'aabb_factor': ${FACTOR}, 'use_lpips': True, 'parallel_sampling': False, 'raymarching_tolerance': ${GAMMA}}" \
    --output-valid ${OUTPUT}/${GAMMA} \

# bash scripts/validate/validate_nerf_test.sh   800x800 200..400 8000 0.01 d