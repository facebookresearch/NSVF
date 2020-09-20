# just for debugging
DATA="maria"
DATASET=/private/home/jgu/data/shapenet/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/
# MODEL=maria_bugfix.single.1024x1024.seq200.v1.p2048.w.shared_nsvf.vs0.2.ss0.025.hyper.cemb256.prune5000.adam.lr_poly.max150000.lr0.001.clip0.0.seed20.ngpu32
MODEL=maria_bugfix2.single.1024x1024.seq200.v1.p1400.w.shared_nsvf.vs0.2.ss0.025.hyper.cemb256.adam.lr_poly.max150000.lr0.001.clip0.0.seed20.ngpu48
TESTPOSE=/private/home/jgu/data/shapenet/${DATA}_test/traj_poses_frames.txt

SAVE=$SAVE/$MODEL
OUTPUT=$SAVE/render

mkdir -p $OUTPUT

ARCH="nsvf_base"
MODEL_PATH=$SAVE/checkpoint_last.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.01)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-camera-poses $TESTPOSE \
    --model-overrides $MODELARGS \
    --render-resolution "1024x1024" \
    --render-output ${OUTPUT} \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"
