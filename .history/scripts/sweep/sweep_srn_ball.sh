ROOT=/private/home/jgu/data/shapenet/
DATA=maria
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn_ball
mkdir -p ${WORK}

ENGINE=~jgu/work/neuralrendering
pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA} \
    --grid "srn_debug4" \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_GEOv4 \
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
