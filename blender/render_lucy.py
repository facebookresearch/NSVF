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
from math import pi
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
    obj_file = args.obj_root + '/' + args.obj_name + '.ply'

if not os.path.exists(render_root):
    os.makedirs(render_root)

prior_objects = [object.name for object in bpy.context.scene.objects]
# do your stl imports here

bpy.context.scene.frame_set(args.frame_id)
#bpy.ops.import_scene.obj(filepath=obj_file)
bpy.ops.import_mesh.ply(filepath=obj_file)

new_current_objects = [object.name for object in bpy.context.scene.objects]
new_objects = set(new_current_objects)-set(prior_objects) 
# obj_name = Path(obj_file).stem
# print([p for p in bpy.data.objects])
# # # hard-coded transformation for imported objects (NO CHANGE)

def color_vertex(obj, color):
    """Paints a single vertex where vert is the index of the vertex
    and color is a tuple with the RGB values."""

    mesh = obj.data 
    scn = bpy.context.scene

    #check if our mesh already has Vertex Colors, and if not add some... (first we need to make sure it's the active object)
    scn.objects.active = obj
    obj.select = True
    if mesh.vertex_colors:
        vcol_layer = mesh.vertex_colors.active
    else:
        vcol_layer = mesh.vertex_colors.new()

    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vcol_layer.data[loop_index].color = color

#example usage
color = (1.0, 0.0, 0.0)  # pink

for obj_name in new_objects:
    obj = bpy.context.scene.objects[obj_name]
    obj.select = True
    print(obj.name)
    obj.scale = (0.0025, 0.0025, 0.0025)
    obj.rotation_euler[0] = pi/2
    obj.location.z -= 1.0
    mat = bpy.data.materials.new('material_1')
    mat.use_vertex_color_paint = True
    obj.active_material = mat

bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = True
lamp2.energy = 0.8

bpy.ops.object.lamp_add(type='POINT')
print([object.name for object in bpy.context.scene.objects])
lamp = bpy.data.lamps['Point']
# lamp2.shadow_method = 'NOSHADOW'
lamp.use_specular = False
lamp.energy = 0.05
bpy.data.objects['Point'].rotation_euler = bpy.data.objects['Sun'].rotation_euler
bpy.data.objects['Point'].rotation_euler[0] += 180

bpy.data.worlds["World"].light_settings.use_ambient_occlusion = True
# bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY')
# obj.rotation_euler = (radians(176.551), radians(31.157), radians(-225.614))
# obj.location.x += -0.21237
# obj.location.y += 0.85433
# obj.location.z += 2.15389
#obj.location.x += -0.65674
#obj.location.y -= 0.54798
#obj.location.z += 2.08499

# Iterate over all members of the material struct
# for item in bpy.data.materials:
#     #Enable "use_shadeless"
#     item.use_shadeless = True

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
