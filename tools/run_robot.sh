DIR="/private/home/jgu/data/shapenet/robot/0000/"
DIR="/private/home/jgu/data/shapenet/robot2/robot/0000/"
python tools/visual_hull.py \
    --dir $DIR --frames 200 --voxel_res 250 --th 120 \
    --fname "voxel2.txt" \
    --image_res "800x800" --extent 10.0 \
    --downsample 0.8 --boundingbox --expand_bbox 0.5 \
