DIR="/private/home/jgu/data/shapenet/vlad_new"
python tools/visual_hull.py \
    --dir $DIR --frames 89 --voxel_res 512 \
    --image_res 2056 --image_res_H 1504 --extent 10.0 \
    --load_mask \
    --downsample 0.1