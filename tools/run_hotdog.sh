DIR="/private/home/jgu/data/shapenet/hotdog2"
python tools/visual_hull.py \
    --dir $DIR --frames 100 --voxel_res 256 \
    --image_res 800 --extent 1.2 \
    --downsample 0.06 --boundingbox