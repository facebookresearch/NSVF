DIR="/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan118/scan118/"
python tools/visual_hull.py \
    --dir $DIR --frames 64 --voxel_res 250 \
    --load_mask --fname "voxel4.txt" \
    --image_res "1200x1600" --extent 2.5 \
    --downsample 0.1 --boundingbox --expand_bbox 2.0 \
