#!/bin/bash
export PATH=$PATH:/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/
export SHELL=$(type -p bash)
render() {
    frameid=`printf "%04d" $3`
    blender --background maria.blend --python render.py -- --obj_root $1 --root $2 --obj_name $3 --scene chair
}

export -f render

DATA=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/models/
OUTPUT=/private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/debug/output/
render $DATA $OUTPUT model_normalized

