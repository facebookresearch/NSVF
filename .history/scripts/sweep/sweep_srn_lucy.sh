ROOT=/private/home/jgu/data/shapenet/lucy3
DATA=lucy3
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn_lucy3
mkdir -p ${WORK}

GRID=srn_shapenet
GRID=srn_shapenet_geo2
ENGINE=~jgu/work/neuralrendering
pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  
python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA}/0001 \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_GEOv42 \
    --num-trials 1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition priority \
    --resume-failed \
    # --local \
    # --dry-run
popd

# python fb_sweep/sweep_neural_rendering.py \
#     --data ${ROOT}/${DATA}/0001 \
#     --grid $GRID \
#     --user-dir "fairnr" \
#     --checkpoints-dir ${WORK} \
#     --no-tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}2_GEOv4 \
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
#     # --dry-run
# popd