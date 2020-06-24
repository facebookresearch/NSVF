ROOT=/private/home/jgu/data/shapenet/bunny
DATA=bunny
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn_bunny
mkdir -p ${WORK}

GRID=srn_shapenet
GRID=srn_shapenet_geo
ENGINE=~jgu/work/neuralrendering
pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA}/0001 \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_SRN \
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
