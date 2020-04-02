DIR="/private/home/jgu/data/shapenet/hotdog"
python tools/visual_hull.py \
    --dir $DIR --frames 100 --voxel_res 400 \
    --image_res 800 --extent 1.2 \
    --downsample 0.03