ROOT=/private/home/jgu/data/shapenet/
DATA=maria
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn
mkdir -p ${WORK}

ENGINE=~jgu/work/neuralrendering
pushd $ENGINE

python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA} \
    --grid "srn_debug4" \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_Geo \
    --num-trials 1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --partition learnfair \
    --comment NeurIPS2020 \
    --resume-failed \
    # --local \
    # --dry-run

popd
