import bpy
import sys
sys.path.append(bpy.path.abspath('//.'))
import bpy_extras
from mathutils import Matrix
import numpy as np
np.set_printoptions(suppress=True)
import time
import shutil
import os
from math import radians
import lib as utils
from pathlib import Path

import sys, argparse, os

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--obj_root', type=str, help='path where obj files can be found')
parser.add_argument('--root', type=str, help='path to save the output')
parser.add_argument('--frame_id', type=int, default=1)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--scene', type=str, default='vlad')
parser.add_argument('--obj_name', type=str, default=None)
parser.add_argument('--resolution', type=int, default='1024')

argv = sys.argv
argv = argv[argv.index("--") + 1:]  # get all args after "--"
args = parser.parse_args(argv)

for scene in bpy.data.scenes:
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution

render_root = '%s/%s/%04d/'%(args.root,args.scene,args.frame_id)

if args.obj_name is None:
    obj_file = args.obj_root + '/Frame_%05d.obj'%args.frame_id
else:
    obj_file = args.obj_root + '/' + args.obj_name + '.obj'

if not os.path.exists(render_root):
    os.makedirs(render_root)

bpy.context.scene.frame_set(args.frame_id)
bpy.ops.import_scene.obj(filepath=obj_file)

# obj_name = Path(obj_file).stem
# print(obj_name)
# print([p for p in bpy.data.objects])
# # hard-coded transformation for imported objects (NO CHANGE)
# obj = bpy.data.objects[obj_name]
# obj.select = True

# bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
# obj.scale = (0.001, 0.001, 0.001)
# obj.rotation_euler = (radians(176.551), radians(31.157), radians(-225.614))
# obj.location.x += -0.21237
# obj.location.y += 0.85433
# obj.location.z += 2.15389
#obj.location.x += -0.65674
#obj.location.y -= 0.54798
#obj.location.z += 2.08499

# Iterate over all members of the material struct
for item in bpy.data.materials:
    #Enable "use_shadeless"
    item.use_shadeless = True

scene = bpy.context.scene
scene.render.alpha_mode = 'TRANSPARENT'

for obj in bpy.data.objects:
    if obj.type == 'CAMERA':
        utils.render(args.frame_id,obj.name,args.root,args.scene)

for scene in bpy.data.scenes:
    scene.render.resolution_x = args.resolution
    scene.render.resolution_y = args.resolution
    

# save intrinsics.txt
#P,K,RT=utils.get_3x4_P_matrix_from_blender(bpy.data.objects['cam_train_0001'])
P,K,RT=utils.get_3x4_P_matrix_from_blender(bpy.data.objects['cam_train_0001'])
intrinsic_txt='''%f %f %f 0.
0. 0. 0.
0.
1.
%d %d
'''%(K[0][0],K[0][2],K[1][2],args.resolution,args.resolution)

f = open('%s/intrinsics.txt'%(render_root), 'w')
f.write(intrinsic_txt)
f.close()
