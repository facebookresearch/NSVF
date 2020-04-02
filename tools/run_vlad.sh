DIR="/private/home/jgu/data/shapenet/vlad1"
python tools/visual_hull.py \
    --dir $DIR --frames 89 --voxel_res 256 --image_res 3008 --extent 5.0 \
    --downsample 0.1