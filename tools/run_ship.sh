DIR="/private/home/jgu/data/shapenet/ship/"
python tools/visual_hull.py \
    --dir $DIR --frames 100 --voxel_res 250 --th 75 \
    --fname "voxel.txt" \
    --image_res "800x800" --extent 7.0 \
    --downsample 0.4 --boundingbox --expand_bbox 0.5 \
