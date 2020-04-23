ROOT=/private/home/jgu/data/
DATA=srn_data
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_chairs
mkdir -p ${WORK}

GRID=geo_shapenet_seq128
ENGINE=~jgu/work/fairdr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
# python fb_sweep/sweep_shapenet.py \
#     --data ${ROOT}/${DATA}/training_set \
#     --grid $GRID \
#     --user-dir "fairdr" \
#     --checkpoints-dir ${WORK} \
#     --tensorboard-logdir ${WORK}/tensorboard/bigbatch/ \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}_128z \
#     --num-trials -1 \
#     --num-gpus 8 \
#     --num-nodes 1 \
#     --mem 500gb \
#     --constraint volta32gb \
#     --exclusive \
#     --comment "NeurIPS2020 deadline." \
#     --partition priority \
#     --resume-failed \
#     --local \
   
# popd

python fb_sweep/sweep_shapenet.py \
    --data ${ROOT}/${DATA}/training_set  \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --no-tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}v2 \
    --num-trials 1 \
    --num-gpus 1 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition priority \
    --resume-failed \
    --local \

popd
# #     # --dry-run
