DIR="/private/home/jgu/data/shapenet/steamtrain/0000/"
python tools/visual_hull.py \
    --dir $DIR --frames 200 --voxel_res 250 --th 120 \
    --fname "voxel2.txt" \
    --image_res "800x800" --extent 22.0 \
    --downsample 1.0 --boundingbox --expand_bbox 1.0 \
