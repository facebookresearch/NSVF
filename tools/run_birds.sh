DIR="/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan106/scan106/"
python tools/visual_hull.py \
    --dir $DIR --frames 64 --voxel_res 160 \
    --load_mask --fname "voxel4.txt" \
    --image_res "1200x1600" --extent 2.0 \
    --downsample 0.16 --boundingbox 