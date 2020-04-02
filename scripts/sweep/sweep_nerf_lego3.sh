ROOT=/private/home/jgu/data/shapenet/
DATA=bulldozer6
WORK=/checkpoint/jgu/space/neuralrendering/debug_nerf_legov2
mkdir -p ${WORK}

GRID=srn_lego
GRID=geo_nerf_lego
GRID=geo_nerf_lego_dev
ENGINE=~jgu/work/neuralrendering
pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_nerf.py \
    --data ${ROOT}/${DATA} \
    --grid $GRID \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_bugfix \
    --num-trials 1 \
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

# python fb_sweep/sweep_nerf.py \
#     --data ${ROOT}/${DATA}  \
#     --grid $GRID \
#     --user-dir "fairdr" \
#     --checkpoints-dir ${WORK} \
#     --no-tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}2_GEOv7 \
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
# #     # --dry-run
# popd