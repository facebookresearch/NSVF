# just for debugging
DATA="wineholder"
RES="800x800"
ARCH="nsvf_base"
DATASET=/private/home/jgu/data/shapenet/${DATA}/scaled2
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/test_$DATA
MODEL_PATH=$SAVE/checkpoint_last.pt

python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views "200..400" \
    --valid-view-resolution "800x800" \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides '{"chunk_size":1024,"raymarching_tolerance":0.01,"tensorboard_logdir":"","eval_lpips":True}' \