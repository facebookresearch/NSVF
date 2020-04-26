ROOT=/private/home/jgu/data/
DATA=srn_data
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_chairs
MODEL=srn_data_test.fp16.seq.128x128.s16.v1.fs64.geo_nerf_transformer.emb384.id.enc1.cac.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max100000.lr0.001.seed2.ngpu8


python validate.py \
    $ROOT/$DATA/testing_set/test.txt \
    --fp16 \
    --user-dir fairdr \
    --task sequence_object_rendering \
    --max-sentences 64 \
    --path ${WORK}/${MODEL}/checkpoint_last.pt \
    --model-overrides "{'reset_context_embed': False, 'valid_views': '0..251'}"