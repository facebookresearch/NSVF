# Download blender 2.79 first
BLENDER=/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/blender
OBJ=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/1a6f615e8b1b5ae4dbbc9440457e303e/models/model_normalized.obj
VIEWS=20
OUTPUT=./tmp

$BLENDER --background --verbose 4 --python render.py -- --output_folder $OUTPUT --views $VIEWS --camera_trace 'sphere_random' $OBJ # > nul 2>&1