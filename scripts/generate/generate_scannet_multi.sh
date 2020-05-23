# just for debugging
ROOT=/private/home/jgu/data/3d_ssl2/ScannetScan/data_render1/
DATA=scene0024_00
DATASET=${ROOT}/${DATA}/data
MODEL_PATH=$1
GPU=${3:-1}

MODEL0=scannet0024_00_rgbd_newx2.single.480x640.s1.v4.geo_nerf.emb384.ss0.0125.v0.1.posemb.sdfh128.raydir.r24.ps.d.p2048.sc0.95.chk512.rgb128.0.depth10.0.ent0.0.adam.lr_poly.max100000.lr0.001.clip0.0.wd0.0.seed20.ngpu8

# CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 1 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution "480x640" \
    --render-camera-poses ${DATASET}/test_traj.txt \
    --render-output /private/home/jgu/data/test_images/output_scannet \
    --render-output-types "rgb" "depth" "normal" \
    --render-combine-output
    
