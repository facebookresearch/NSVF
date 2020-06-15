DIR="/private/home/jgu/data/shapenet/ignatius_srn"
python tools/visual_hull.py \
    --dir $DIR --frames 263 --voxel_res 256 \
    --load_mask --fname "voxel0.5txt" \
    --image_res "1080x1920" --extent 8.0 \
    --downsample 0.5 --boundingbox \
    --th 136