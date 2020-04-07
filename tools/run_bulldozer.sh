DIR="/private/home/jgu/data/shapenet/bulldozerbbox"
python tools/visual_hull.py \
    --dir $DIR --frames 100 --voxel_res 256 --image_res 800 --extent 5.0 \
    --downsample 0.24 --boundingbox