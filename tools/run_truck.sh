DIR="/private/home/jgu/data/shapenet/truck/"
python tools/visual_hull.py \
    --dir $DIR --frames 250 --voxel_res 200 --th 150 \
    --fname "voxel2.txt" \
    --image_res "1080x1920" --extent 8.0 \
    --downsample 0.2 --boundingbox --expand_bbox 1.2 \
