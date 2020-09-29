# just for debugging
DATA="Jade"
RES="576x768"
ARCH="nsvf_base"
DATASET=/private/home/jgu/data/shapenet/release/BlendedMVS/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA
MODEL_PATH=$SAVE/$ARCH/checkpoint_last.pt

# additional rendering args
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.0)

python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses $DATASET/test_traj.txt \
    --model-overrides $MODELARGS \
    --render-resolution $RES \
    --render-output ${SAVE}/$ARCH/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"