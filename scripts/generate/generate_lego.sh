# just for debugging
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/61b984febe54b752d61420a53a0cb96d
# DATASET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/debug
DATA=${2:-"bulldozer2"}
DATASET=/private/home/jgu/data/shapenet/bulldozer
DATASET=/private/home/jgu/data/shapenet/${DATA}

# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn_ref21
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/slurm_srn/maria.no_c10d.single.512x512.s1.v5.p16384.mask0.75.march10.rgb200.dep0.08.vgg1.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/slurm_srn/maria.no_c10d.single.512x512.s1.v5.p16384.mask0.5.bbox.march10.rgb200.dep0.08.vgg1.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
MODEL_PATH=/checkpoint/jgu/space/neuralrendering/slurm_srn/maria.no_c10d.single.512x512.s1.v5.p16384.mask0.5.bbox.march10.rgb200.dep0.08.vgg1.0.srn_base.adam.poly.lr0.001.clip0.0.wd0.0.seed2.dwd.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn/61b984febe54b752d61420a53a0cb96d_inf.no_c10d.single.512x512.s1.v5.p30000.mask0.0.march10.rgb200.dep1.0.vgg0.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn/61b984febe54b752d61420a53a0cb96d_inf.no_c10d.single.128x128.s1.v5.march10.rgb200.dep1.0.vgg0.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn/debug_inf2.no_c10d.single.512x512.s1.v5.p30000.mask0.0.march10.rgb200.dep1.0.vgg0.0.srn_base.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
# MODEL_PATH=/checkpoint/jgu/space/neuralrendering/debug_srn/maria_inf2.no_c10d. single.512x512.s1.v5.march10.geo.rgb200.dep0.08.vgg1.0.geosrn_simple.adam.lr_fixed.lr0.001.clip0.0.wd0.0.seed2.ngpu8
MODEL_PATH=$1
GPU=${3:-0}

CUDA_VISIBLE_DEVICES=${GPU} \
python render.py ${DATASET} \
    --user-dir fairdr \
    --task single_object_rendering \
    --load-point \
    --path ${MODEL_PATH}/checkpoint_last.pt \
    --render-beam 10 \
    --render-angular-speed 3 \
    --render-save-fps 24 \
    --render-num-frames 120 \
    --render-resolution 400 \
    --render-path-args "{'radius': 3.5, 'h': 1.5, 'axis': 'z', 't0': -2, 'r':-1}" \
    --render-output /private/home/jgu/data/test_images/output3 \
    --render-output-types "rgb"  \

#  --render-path-args "{'radius': 3.5, 'h': 0.0, 'axis': 'z'}" \
#     --render-up-vector "(0,0,-1)" \