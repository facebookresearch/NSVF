#!/bin/bash
export PATH=$PATH:/private/home/jgu/tools/blender-2.79b-linux-glibc219-x86_64/
export SHELL=$(type -p bash)
render() {
    frameid=`printf "%04d" $3`
    /private/home/jgu/tools/blender-2.82a-linux64/blender --background new.blend --python render_bulldozer.py -- --root $1 --scene statue
}

export -f render

OUTPUT=/private/home/jgu/data/shapenet/statue
mkdir -p $OUTPUT

render $OUTPUT