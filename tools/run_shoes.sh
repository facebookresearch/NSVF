DIR="/private/home/jgu/data/shapenet/shoes/"
python tools/visual_hull.py \
    --dir $DIR --frames 28 --voxel_res 250 --th 14 \
    --fname "voxel2.txt" --load_mask \
    --image_res "576x768" --extent 7.0 \
    --downsample 0.4 --boundingbox --expand_bbox 0.5 \
