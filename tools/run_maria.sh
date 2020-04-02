DIR="/private/home/jgu/data/shapenet/maria3"
python tools/visual_hull.py \
    --dir $DIR --frames 50 --load_mask --voxel_res 256 --image_res 1024 --extent 5.0 \
    --downsample 0.07