ROOT=/private/home/jgu/data/shapenet/
DATA=multi_nerf
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_full2
mkdir -p ${WORK}

GRID=geo_nerf
# GRID=geo_nerf0
ENGINE=~jgu/work/fairnr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_multinerf.py \
    --data ${ROOT}/${DATA}/train_full.txt  \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${GRID}try \
    --num-trials -1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 supplemental deadline." \
    --partition priority \
    --resume-failed \
    --local \
   
# popd

# python fb_sweep/sweep_babyangel.py \
#     --data ${ROOT}/${DATA}/  \
#     --grid $GRID \
#     --user-dir "fairnr" \
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
