DIR="/private/home/jgu/data/shapenet/differentiable_volumetric_rendering/data/DTU/scan106/scan106/"
python tools/visual_hull.py \
    --dir $DIR --frames 64 --voxel_res 128 \
    --load_mask \
    --image_res 1600 --image_res_H 1200 --extent 2.0 \
    --downsample 0.1 --boundingbox 