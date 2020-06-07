MODEL_ROOT=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3/
DATA=${1}
MODEL=${2}
RES=${3:-"800x800"}
VALID=${4:-"200..400"}
FACTOR=${5:-6400}
GAMMA=${6:-0}
OUTPUT=$MODEL_ROOT/test_output/${MODEL}

mkdir -p  ${OUTPUT}/${GAMMA}

# export CUDA_VISIBLE_DEVICES=0
python validate.py \
    $DATA \
    --valid-views ${VALID} \
    --valid-view-resolution ${RES} \
    --user-dir fairdr \
    --no-preload \
    --task single_object_rendering \
    --max-sentences 1 \
    --valid-view-per-batch 1 \
    --path ${MODEL_ROOT}/${MODEL}/checkpoint_last.pt \
    --model-overrides "{'chunk_size': 512, 'aabb_factor': ${FACTOR}, 'use_lpips': True, 'parallel_sampling': True, 'raymarching_tolerance': ${GAMMA}}" \
    --output-valid ${OUTPUT}/${GAMMA}v5 \


    #