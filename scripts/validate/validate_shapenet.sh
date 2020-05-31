ROOT=/private/home/jgu/data/
DATA=srn_data
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_chairsv3
MODEL=srn_data_128z.fp16.seq.128x128.s16.v1.geo_nerf_transformer.emb384.id.enc1.cac.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.clip0.0.wd0.0.seed2.ngpu8
MODEL=srn_data_pos.fp16.seq.128x128.s16.v1.geo_nerf.hpc.emb384.id.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.p2048.smk0.9.patch4.chk512.rgb128.0.alpha0.0.latent0.0.vgg0.0.adam.lr_poly.max200000.lr0.001.seed2.ngpu8
MODEL=srn_data_biglr.seq.128x128.s16.v1.geo_nerf.qxyz.hyper.emb256.nf3.nt4.id.ss0.025.v0.25.posemb.sdfh128.bg1.0.dis.ps.pruning.dyvox.maxp.th\{val\}.p512.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max100000.lr0.001.seed22.ngpu8
# MODEL=srn_data_biglr.fp16.seq.128x128.s16.v1.geo_nerf.qxyz.hpc.emb256.nf3.nt4.id.ss0.025.v0.25.posemb.sdfh128.bg1.0.dis.ps.pruning.dyvox.maxp.th\{val\}.p512.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max100000.lr0.001.seed22.ngpu8
# MODEL=srn_data_nparea.fp16.seq.128x128.s8.v1.geo_nerf.qxyz.hpc.emb256.nf3.nt4.id.ss0.025.v0.25.posemb.sdfh128.pos72.bg1.0.dis.ps.p512.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max100000.lr0.001.seed22.ngpu8
# MODEL=srn_data_areaz.fp16.seq.128x128.s8.v1.geo_nerf.hpc.emb256.nf3.nt4.id.ss0.025.v0.25.posemb.sdfh128.bg1.0.dis.ps.pruning.dyvox.maxp.th\{val\}.p512.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max100000.lr0.001.seed22.ngpu8

#CUDA_VISIBLE_DEVICES=0 \
python validate.py \
    $ROOT/$DATA/training_set/test_small.txt \
    --object-id-path $ROOT/$DATA/training_set/object_ids.txt \
    --user-dir fairdr \
    --task sequence_object_rendering \
    --max-sentences 50 \
    --path ${WORK}/${MODEL}/checkpoint_best.pt \
    --model-overrides "{'reset_context_embed': False, 'subsample_valid': 1, 'valid_views': '0..50'}"\
    --output-valid $ROOT/test_images/output20/ \

#  --model-overrides "{'total_num_embedding': 12000, 'reset_context_embed': False, 'subsample_valid': 1, 'valid_views': '0..50'}"\