DIR="/private/home/jgu/data/shapenet/lighthouse/"
python tools/visual_hull.py \
    --dir $DIR --frames 200 --voxel_res 200 --th 175 \
    --fname "voxel3.txt" \
    --image_res "1024x1024" --extent 20.0 \
    --downsample 1.0 --boundingbox --expand_bbox 0.5 \
