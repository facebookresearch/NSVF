ROOT=/private/home/jgu/data/shapenet/maria
DATA=maria_seq_small
WORK=/checkpoint/jgu/space/neuralrendering/debug_srn
mkdir -p ${WORK}

ENGINE=~jgu/work/neuralrendering
pushd $ENGINE

python fb_sweep/sweep_neural_rendering.py \
    --data ${ROOT}/${DATA}.txt \
    --grid "srn_debug_seq2" \
    --user-dir "fairdr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard/transformer_seq_small \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_Transformerv2 \
    --num-trials -1 \
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
