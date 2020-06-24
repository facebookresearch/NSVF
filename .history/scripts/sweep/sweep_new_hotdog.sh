ROOT=/private/home/jgu/data/shapenet/
DATA=nerf_hotdog
#DATA=nerf_hotdog2
WORK=/checkpoint/jgu/space/neuralrendering/debug_new_hotdog
mkdir -p ${WORK}

GRID=geo_new_hotdog
#GRID=geo_new_hotdog2
ENGINE=~jgu/work/neuralrendering
pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \

python fb_sweep/sweep_nerf.py \
    --data ${ROOT}/${DATA} \
    --grid $GRID \
    --user-dir "fairnr" \
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
    --partition learnfair \
    --resume-failed \
    --local \
   
# popd

# python fb_sweep/sweep_nerf.py \
#     --data ${ROOT}/${DATA}  \
#     --grid $GRID \
#     --user-dir "fairnr" \
#     --checkpoints-dir ${WORK} \
#     --no-tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}2_GEOv7 \
#     --num-trials 1 \
#     --num-gpus 8 \
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