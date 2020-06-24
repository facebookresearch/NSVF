ROOT=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/
DATA=61b984febe54b752d61420a53a0cb96d
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn
mkdir -p ${WORK}

ENGINE=~jgu/work/neuralrendering
pushd $ENGINE

python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA} \
    --grid "srn_shapenet" \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_inf2 \
    --num-trials 1 \
    --num-gpus 8 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "ECCV paper deadline." \
    --partition priority \
    --resume-failed \
    --local \
    # --dry-run

popd
