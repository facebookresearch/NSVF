# just for debugging
MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_scannetv2/select_0241nodepth.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_scannetv2/select_0241v2.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent10.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_scannetv2/select_0241v2.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent0.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_scannetv2/select_0241nodepth.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent0.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_scannetv2/select_0241_bboxv2.single.480x640.s1.v4.geo_nerf.emb384.ss0.05.v0.4.posemb.sdfh128.raydir.r24.spec.sd0.5.dis.ps.maxp.dyvox.80k.d.p2048.sc0.95.chk512.rgb128.0.depth0.0.ent0.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8
DATASET=/private/home/jgu/data/shapenet/select_0241
TESTTRAJ=/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0241_01/data/test_traj/test_traj.txt
TESTTRAJ=/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/scene0241_01/pose_manual_v2_skipped
FACTOR=${2:-8192}
GAMMA=${3:-0.01}

# CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 40 \
    --render-num-frames 100 \
    --render-resolution "512x512" \
    --render-camera-poses ${TESTTRAJ} \
    --render-camera-intrinsics /private/home/jgu/data/shapenet/scannet/data_render2/scene0101_04/data/intrinsics.txt \
    --render-output ${MODEL_ROOT}/supplemental/scannet241_v2v2/ \
    --model-overrides "{'fp16': False, 'aabb_factor': ${FACTOR}, 'raymarching_tolerance': ${GAMMA}}" \
    --render-output-types "rgb" "depth" "hit" "normal" \
    --render-combine-output
    
