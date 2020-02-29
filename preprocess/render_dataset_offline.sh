# Download blender 2.79 at https://www.blender.org/download/releases/2-79/
BLENDER=/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/blender
TARGET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug
VIEWS=100
OUTPUT=$TARGET
LOGFILE=$(mktemp)

pushd blender

find $TARGET -name *.obj -print0 | \
    xargs -0 -n1 -P40 -I {} \
    $BLENDER --background --python offline_render.py -- --output_folder $TARGET --views $VIEWS --camera_trace 'sphere_random' {} #> $LOGFILE 2>&1

popd