ROOT=/private/home/jgu/data/srn_data/
DATA=train
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_chairs
mkdir -p ${WORK}

GRID=geo_shapenet_seq
ENGINE=~jgu/work/fairnr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
# python fb_sweep/sweep_shapenet.py \
#     --data ${ROOT}/${DATA}.txt \
#     --grid $GRID \
#     --user-dir "fairnr" \
#     --checkpoints-dir ${WORK} \
#     --tensorboard-logdir ${WORK}/tensorboard/bigbatch/ \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}_bigbatch \
#     --num-trials -1 \
#     --num-gpus 8 \
#     --num-nodes 4 \
#     --mem 500gb \
#     --constraint volta32gb \
#     --exclusive \
#     --comment "NeurIPS2020 deadline." \
#     --partition learnfair \
#     --resume-failed \
    # --local \
   
# popd

python fb_sweep/sweep_shapenet.py \
    --data ${ROOT}/${DATA}.txt  \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --no-tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}v10 \
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
