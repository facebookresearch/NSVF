ROOT=/private/home/jgu/data/shapenet
# DATA=lego_full
# DATA=wine-holder
# DATA=mic_full
# DATA=chair_full
# DATA=wineholder/0000
DATA=final/blendedmvs_character

MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_drumsv2.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_ficusv2.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_drums_reloadv2.1_reload.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_legov2.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2/geo_wineholderv1.fp16.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_wineholdervfine.single.800x800.s1.v4.geo_nerf.emb384.ss0.025.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_mic2vhp2.single.800x800.s1.v4.geo_nerf.emb32.ss0.05.v0.4.posemb.sdfh128.raydir.r24.addpos6.bg1.0.bgsg.dis.prune2500.th0.5.dy2.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_micv1.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_chairv1.single.800x800.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p2048.chk512.rgb128.0.alpha1.0.vgg1.0.l3.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_wineholder2_latest.single.800x800.s1.v4.geo_nerf.emb32.addpos6.ss0.025.v0.2.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.maxp.60k.p16384.chk512.rgb128.0.alpha1.0.adam.lr_poly.max150000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
MODEL=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/geo_character_finalv1.single.576x768.s1.v4.geo_nerf.emb32.addpos6.ss0.01.v0.08.posemb.sdfh128.raydir.r24.bg1.0.bgsg.dis.prune2500.th0.5.dyvox.100k.p2048.chk512.rgb128.0.alpha1.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8/
# ssi 0.921625 | psn 24.6835


mkdir -p  $ROOT/test_images/output_nsvf/

# CUDA_VISIBLE_DEVICES=0 \
python validate.py \
    $ROOT/$DATA \
    --valid-views "51..59" \
    --valid-view-resolution "576x768" \
    --user-dir fairdr \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL}/checkpoint_best.pt \
    --model-overrides "{'chunk_size': 1024, 'aabb_factor': 8192}"

    #  --output-valid $ROOT/test_images/output_nsvf/ \
python validate.py \
    $ROOT/$DATA \
    --valid-views "51..59" \
    --valid-view-resolution "576x768" \
    --user-dir fairdr \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL}/checkpoint_best.pt \
    --model-overrides "{'chunk_size': 1024, 'aabb_factor': 8192, 'raymarching_tolerance': 0.01}"