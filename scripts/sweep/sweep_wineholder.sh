ROOT=/private/home/jgu/data/shapenet/
DATA=wineholder
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3
mkdir -p ${WORK}

GRID=geo_wineholder4
ENGINE=~jgu/work/fairdr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_wineholder.py \
    --data ${ROOT}/${DATA}/0000  \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard/ \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${GRID}_latest_bv0 \
    --num-trials -1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition priority \
    --resume-failed \
    # --dry-run
    # --local \
   
# popd
# python fb_sweep/sweep_babyangel.py \
#     --data ${ROOT}/${DATA}/  \
#     --grid $GRID \
#     --user-dir "fairdr" \
#     --checkpoints-dir ${WORK} \
#     --no-tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}v9 \
#     --num-trials 1 \
#     --num-gpus 1 \
#     --num-nodes 1 \
#     --mem 500gb \
#     --constraint volta32gb \
#     --exclusive \
#     --comment "NeurIPS2020 deadline." \
#     --partition priority \
#     --resume-failed \
#     --local \

popd
# #     # --dry-run
