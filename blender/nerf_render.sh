
render() {
     echo "find ${1}"
     /private/home/jgu/tools/blender-2.82a-linux64/blender \
          --background /private/home/jgu/data/shapenet/shapenet_chair/bg.blend \
          --python nerf_render_obj.py -- --model $1
}
export -f render

# find . -name '*.html' | parallel gzip --best 
find /private/home/jgu/data/shapenet/ShapeNetCore.v2/03001627/*/models/model_normalized.obj | parallel -j 8 render