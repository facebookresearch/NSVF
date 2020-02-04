# Download blender 2.79 first
BLENDER=/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/blender
TARGET=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627
VIEWS=100
OUTPUT=$TARGET

find /private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/  -name *.obj -print0 | \
    xargs -0 -n1 -P40 -I {} \
    $BLENDER --background --python render.py -- --output_folder $TARGET --views $VIEWS --camera_trace 'sphere_random' {}  > nul 2>&1