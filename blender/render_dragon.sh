#!/bin/bash
export PATH=$PATH:/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/
export SHELL=$(type -p bash)
render() {
    blender --background maria.blend --python render_lucy.py -- --obj_root $1 --root $2 --obj_name $3 --scene $3
}

export -f render

# DATA=/private/home/jgu/data/shapenet/bunny
# OUTPUT=/private/home/jgu/data/shapenet/bunny
# render $DATA $OUTPUT bunny

DATA=/private/home/jgu/data/shapenet/lucy3
OUTPUT=/private/home/jgu/data/shapenet/lucy3
render $DATA $OUTPUT lucy3