ROOT=/private/home/jgu/data/3d_ssl2/ScannetScan/data_render2/
DATA=scene0049_00
WORK=/checkpoint/jgu/space/neuralrendering/debug_scannetv2
mkdir -p ${WORK}

# GRID=geo_scannet00
# GRID=geo_scannet01
GRID=geo_scannet2
ENGINE=~jgu/work/fairnr-exp

pushd $ENGINE
#  --tensorboard-logdir ${WORK}/tensorboard \
#  --tensorboard-logdir ${WORK}/tensorboard \
python fb_sweep/sweep_scannet2.py \
    --data ${ROOT}/${DATA}/data  \
    --grid $GRID \
    --user-dir "fairnr" \
    --checkpoints-dir ${WORK} \
    --tensorboard-logdir ${WORK}/tensorboard \
    --snapshot-code \
    --snapshot-root ${WORK}/snapshot \
    --prefix ${DATA}v3 \
    --num-trials -1 \
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

# CUDA_VISIBLE_DEVICES=0 \
# python fb_sweep/sweep_scannet.py \
#     --data ${ROOT}/${DATA}/data  \
#     --grid $GRID \
#     --user-dir "fairnr" \
#     --checkpoints-dir ${WORK} \
#     --tensorboard-logdir ${WORK}/tensorboard \
#     --snapshot-code \
#     --snapshot-root ${WORK}/snapshot \
#     --prefix ${DATA}v9 \
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

popd
# #     # --dry-run
