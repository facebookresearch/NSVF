DIR="/private/home/jgu/data/shapenet/wineholder/0000/"
python tools/visual_hull.py \
    --dir $DIR --frames 200 --voxel_res 256 --th 150 \
    --fname "voxel0.2.txt" \
    --image_res "800x800" --extent 10.0 \
    --downsample 0.2 --boundingbox --expand_bbox 1.0 \
