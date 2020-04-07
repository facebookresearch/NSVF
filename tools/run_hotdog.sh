DIR="/private/home/jgu/data/shapenet/nerf_hotdog2"
python tools/visual_hull.py \
    --dir $DIR --frames 100 --voxel_res 378 \
    --image_res 800 --extent 5.0 \
    --downsample 0.32 --boundingbox 
    # --boundingbox