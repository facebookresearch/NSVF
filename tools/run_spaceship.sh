DIR="/private/home/jgu/data/shapenet/spaceship/"
python tools/visual_hull.py \
    --dir $DIR --frames 400 --voxel_res 128 --th 220 \
    --fname "voxel3.txt" \
    --image_res "800x800" --extent 10.0 \
    --downsample 0.6 --boundingbox --expand_bbox 1.5 \
