# just for debugging
DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627
FAIRDR=fairdr/
FAIRSEQ=/private/home/jgu/work/fairseq-master2
MODEL_PATH=/checkpoint/jgu/space/neuralrendering

mkdir -p $MODEL_PATH


CUDA_VISIBLE_DEVICES=0 \
fairseq-train $DATASET \
    --user-dir fairdr/ \
    --save-dir $MODEL_PATH \
    --max-epoch 80 \
    --task single_object_rendering \
    --arch transformer \
    --log-format json \
    --log-interval 1 \
    --criterion cross_entropy_acc \
