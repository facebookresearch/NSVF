# DIR="/private/home/jgu/data/shapenet/shapenet_chair/render_256/9d36bf414dde2f1a93a28cbb4bfc693b"
# DIR="/private/home/jgu/data/srn_data/chairs_2.0_train/9d36bf414dde2f1a93a28cbb4bfc693b"
visual_hull() {
DIR=$1
python tools/visual_hull.py \
    --dir $DIR --frames 50 --voxel_res 100 --image_res 512 --extent 3.0 \
    --white-bg \
    --downsample 0.25 --visualhull_only
}
export -f visual_hull
parallel -j 66 --linebuffer time visual_hull ::: `find  /private/home/jgu/data/srn_data/chairs_2.0_train/ -maxdepth 1 -name "*"`
