# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# just for debugging
DATA="Wineholder"
RES="800x800"
ARCH="nsvf_base"
SUFFIX="v1"
DATASET=/private/home/jgu/data/shapenet/release/Synthetic_NSVF/${DATA}
SAVE=/checkpoint/jgu/space/neuralrendering/new_release/$DATA
MODEL=$ARCH$SUFFIX
MODEL_PATH=$SAVE/$MODEL/checkpoint_last.pt

# CUDA_VISIBLE_DEVICES=0 \
python render.py ${DATASET} \
    --user-dir fairnr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \
    --render-beam 1 \
    --render-save-fps 24 \
    --render-camera-poses ${DATASET}/pose \
    --render-views "200..400" \
    --model-overrides '{"chunk_size":256,"raymarching_tolerance":0.01}' \
    --render-resolution $RES \
    --render-output ${SAVE}/output \
    --render-output-types "color" "depth" "voxel" "normal" \
    --render-combine-output --log-format "simple"