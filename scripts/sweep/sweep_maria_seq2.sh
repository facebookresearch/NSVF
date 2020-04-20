ROOT=/private/home/jgu/data/shapenet/maria/
DATA=maria_seq_small
DATA=maria_seq
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_mariaseq2
mkdir -p ${WORK}

GRID=geo_maria_seq_reload
GRID=geo_maria_seq2_dyn
GRID=geo_maria_seq_transformer
ENGINE=~jgu/work/fairdr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
# python fb_sweep/sweep_maria.py \
#     --data ${ROOT}/${DATA}.txt \
#     --grid $GRID \
#     --user-dir "fairdr" \
#     --checkpoints-dir ${WORK} \
#     --tensorboard-logdir ${WORK}/tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}_TRAMv7 \
#     --num-trials -1 \
#     --num-gpus 8 \
#     --num-nodes 1 \
#     --mem 500gb \
#     --constraint volta32gb \
#     --exclusive \
#     --comment "NeurIPS2020 deadline." \
#     --partition learnfair \
#     --resume-failed \
    # --local \
   
# popd

python fb_sweep/sweep_maria.py \
    --data ${ROOT}/${DATA}.txt  \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --no-tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_Trans9 \
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
