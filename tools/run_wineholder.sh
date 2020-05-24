DIR="/private/home/jgu/data/shapenet/wine-holder/"
python tools/visual_hull.py \
    --dir $DIR --frames 400 --voxel_res 128 --th 390 \
    --fname "voxel.txt" \
    --image_res "800x800" --extent 4.0 \
    --downsample 0.4 --boundingbox --expand_bbox 0.5 \
