ROOT=/private/home/jgu/data/shapenet/final
DATA=tanksandtemple_barnv2
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_singlev3
mkdir -p ${WORK}

GRID=geo_barn_final
GRID=geo_barn
ENGINE=~jgu/work/fairnr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \

python fb_sweep/sweep_barn.py \
    --data ${ROOT}/${DATA}/  \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard/ \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${GRID}_v1model \
    --num-trials -1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition priority \
    --resume-failed \
    --local \


# python fb_sweep/sweep_ignatius.py \
#     --data ${ROOT}/${DATA}/  \
#     --grid $GRID \
#     --user-dir "fairnr" \
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