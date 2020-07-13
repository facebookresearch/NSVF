ROOT=/private/home/jgu/data/shapenet/maria
DATA=0000
WORK=/checkpoint/jgu/space/neuralrendering/debug_scannet0
mkdir -p ${WORK}

GRID=geo_maria
ENGINE=~jgu/work/fairnr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
# python fb_sweep/sweep_maria.py \
#     --data ${ROOT}/${DATA} \
#     --grid $GRID \
#     --user-dir "fairnr" \
#     --checkpoints-dir ${WORK} \
#     --tensorboard-logdir ${WORK}/tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}_Trans \
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
   
# popd

python fb_sweep/sweep_maria.py \
    --data ${ROOT}/${DATA}  \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}_test0 \
    --num-trials 1 \
    --num-gpus 1 \
    --num-nodes 1 \
    --mem 500gb \
    --constraint volta32gb \
    --exclusive \
    --comment "NeurIPS2020 deadline." \
    --partition priority \
    --resume-failed \
    --local \
# #     # --dry-run
popd