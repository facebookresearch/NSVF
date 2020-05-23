ROOT=/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan65/
DATA=scan65
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_singlev2
mkdir -p ${WORK}

GRID=geo_skull
ENGINE=~jgu/work/fairdr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_skull.py \
    --data ${ROOT}/${DATA}/  \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard/ \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${GRID}sp \
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
