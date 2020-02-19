# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
DATASET=/private/home/jgu/data/shapenet/maria
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn_ref18

CUDA_VISIBLE_DEVICES=0 \
fairseq-generate ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH} \