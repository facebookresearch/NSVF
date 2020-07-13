# just for debugging
ROOT=/private/home/jgu/data/
DATA=srn_data

MODEL_PATH="/checkpoint/jgu/space/neuralrendering/debug_new_chairsv3/srn_data_xyz.fp16.seq.128x128.s16.v1.geo_nerf.qxyz.hpc.emb384.id.ss0.05.v0.25.posemb.sdfh128.bg1.0.dis.ps.pruning.cm.dyvox.fv.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.seed22.ngpu8"
MODEL_PATH="/checkpoint/jgu/space/neuralrendering/debug_new_chairsv2/srn_data_v5.fp16.seq.128x128.s16.v1.geo_nerf.hpc.emb384.id.ss0.03125.v0.0625.posemb.sdfh128.bg1.0.dis.dstep.pruning.cm.dyvox.fv.p2048.smk0.9.chk512.rgb128.0.alpha1.0.latent1.0.vgg0.0.adam.lr_poly.max200000.lr0.001.seed2.ngpu8"
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${ROOT}/${DATA}/training_set/test.txt \
    --user-dir fairnr \
    --task single_object_rendering \
    --test-views "25" \
    --no-preload \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 1 \
    --max-sentences 10 \
    --render-angular-speed 0 \
    --render-save-fps 1 \
    --render-num-frames 25 \
    --render-resolution "128x128" \
    --render-path-args "{'radius': 3.0, 'h': 0.5, 'axis': 'z', 't0': 2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output6 \
    --render-output-types "rgb" "hit" "normal" \
    --render-combine-output
    # "hit" "normal" "depth"
    