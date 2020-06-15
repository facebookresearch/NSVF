DIR="/private/home/jgu/data/shapenet/hotdog_full/"
python tools/visual_hull.py \
    --dir $DIR --frames 200 --voxel_res 250 --th 150 \
    --fname "voxel.txt" \
    --image_res "800x800" --extent 5.0 \
    --downsample 0.4 --boundingbox --expand_bbox 0.5 \
