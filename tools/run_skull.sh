DIR="/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan65/scan65/"
python tools/visual_hull.py \
    --dir $DIR --frames 49 --voxel_res 250 --th 26 \
    --load_mask --fname "bbvoxel0.2.txt" \
    --image_res "1200x1600" --extent 3.0 \
    --downsample 0.2 --boundingbox --expand_bbox 0.5 \
