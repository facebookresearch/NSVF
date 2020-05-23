ROOT=/private/home/jgu/data/shapenet
DATA=drums
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_drumsv2.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/

# ssi 0.921625 | psn 24.6835

python validate.py \
    $ROOT/$DATA \
    --valid-views "200..400" \
    --valid-view-resolution "800x800" \
    --fp16 \
    --user-dir fairdr \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL}/checkpoint_best.pt \
    # --model-overrides "{'total_num_embedding': 12000, 'reset_context_embed': False, 'subsample_valid': 1, 'valid_views': '0..251'}"