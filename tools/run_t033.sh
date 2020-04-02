DIR="/private/home/jgu/data/shapenet/t0332"
python tools/visual_hull.py \
    --dir $DIR --frames 32 --voxel_res 300 \
    --image_res 1024 --image_res_H 768 --extent 5.0 \
    --downsample 0.14