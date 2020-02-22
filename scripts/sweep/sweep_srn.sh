ROOT=/private/home/jgu/data/shapenet/
DATA=maria
WORK=/checkpoint/jgu/space/neuralrendering/slurm_srn/
mkdir -p ${WORK}

ENGINE=~jgu/work/neuralrendering
pushd $ENGINE

python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA} \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --prefix ${DATA} \
    --num-trials 1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "ECCV paper deadline." \
    --partition priority \
    --resume-failed \
    # --local
    # --dry-run

popd
