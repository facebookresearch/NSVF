DATA="wineholder"
DATASET=/private/home/jgu/data/shapenet/${DATA}/0000
SAVE=/checkpoint/jgu/space/neuralrendering/new_codebase/model6
SAVE=/checkpoint/jgu/space/neuralrendering/new_test/wineholder_high_bugfix.single.800x800.v1.p2048.w.nsvf_base.ss0.003125.prune2500.adam.lr_poly.max150000.lr0.001.clip0.0.seed20.ngpu32
MODEL_PATH=$SAVE/checkpoint_best.pt
MODELTEMP='{"chunk_size":%d,"raymarching_tolerance":%.3f,"tensorboard_logdir":"","eval_lpips":True}'
MODELARGS=$(printf "$MODELTEMP" 1024 0.001)

RES="800x800"
VALID=${1:-"200..400"}
OUTPUT=${SAVE}/evalw
mkdir -p  ${OUTPUT}

export CUDA_VISIBLE_DEVICES=0 
python validate.py ${DATASET} \
    --user-dir fairnr \
    --valid-views ${VALID} \
    --valid-view-resolution ${RES} \
    --no-preload \
    --load-depth \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_PATH} \
    --model-overrides $MODELARGS \
    --output-valid ${OUTPUT} \
    | tee -a ${SAVE}/eval.log

