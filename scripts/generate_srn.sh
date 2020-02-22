# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
DATASET=/private/home/jgu/data/shapenet/maria
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn_ref21
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/slurm_srn/maria.no_c10d.single.512x512.s1.v5.p16384.mask0.75.march10.rgb200.dep0.08.vgg1.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8

CUDA_VISIBLE_DEVICES=0 \
fairdr-render ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --path ${MODEL_PATH}/checkpoint_last.pt \