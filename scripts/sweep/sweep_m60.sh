ROOT=/private/home/jgu/data/shapenet/
DATA=m60_srn
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_single
mkdir -p ${WORK}

GRID=geo_tank
# GRID=geo_ignatius
# GRID=geo_ignatius2
# GRID=geo_ignatius_bg
ENGINE=~jgu/work/fairdr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \

python fb_sweep/sweep_tank.py \
    --data ${ROOT}/${DATA}/  \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard/ \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${GRID}v2pcd \
    --num-trials -1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition learnfair \
    --resume-failed \
    --local \


# python fb_sweep/sweep_ignatius.py \
#     --data ${ROOT}/${DATA}/  \
#     --grid $GRID \
#     --user-dir "fairdr" \
#     --checkpoints-dir ${WORK} \
#     --no-tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${GRID}_bb \
#     --num-trials -1 \
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