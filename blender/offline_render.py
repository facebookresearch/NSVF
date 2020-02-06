# A simple script that uses blender to render views of a single object by rotation the camera around it.
# Also produces depth map at the same time.
#
# Example:
# blender --background --python mytest.py -- --views 10 /path/to/my.obj
#

import argparse, sys, os
import numpy as np
import bpy

from math import radians
from utils import get_calibration_matrix_K_from_blender, get_3x4_RT_matrix_from_blender, matrix2str

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('obj', type=str,
                    help='Path to the obj file to be rendered.')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')
parser.add_argument('--remove_shadow', action='store_true', 
                    help='remove shadow for the lamp')
parser.add_argument('--camera_radius', type=float, default=2)
parser.add_argument('--camera_position', type=str, default='(0, 1, 0.6)')
parser.add_argument('--camera_trace', choices=['sphere_random', 'fixed', 'circle'], type=str)

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
bpy.context.scene.render.image_settings.file_format = args.format
bpy.context.scene.render.image_settings.color_depth = args.color_depth

# Clear default nodes
for n in tree.nodes:
    tree.nodes.remove(n)

# Create input render layer node.
render_layers = tree.nodes.new('CompositorNodeRLayers')

scale_normal = tree.nodes.new(type="CompositorNodeMixRGB")
scale_normal.blend_type = 'MULTIPLY'
# scale_normal.use_alpha = True

scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
bias_normal.blend_type = 'ADD'
# bias_normal.use_alpha = True

bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
links.new(scale_normal.outputs[0], bias_normal.inputs[1])

normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
normal_file_output.label = 'Normal Output'
links.new(bias_normal.outputs[0], normal_file_output.inputs[0])

# Delete default cube
bpy.data.objects['Cube'].select = True
bpy.ops.object.delete()

bpy.ops.import_scene.obj(filepath=args.obj)
for object in bpy.context.scene.objects:
    if object.name in ['Camera', 'Lamp']:
        continue
    bpy.context.scene.objects.active = object
    if args.scale != 1:
        bpy.ops.transform.resize(value=(args.scale,args.scale,args.scale))
        bpy.ops.object.transform_apply(scale=True)
    if args.remove_doubles:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles()
        bpy.ops.object.mode_set(mode='OBJECT')
    if args.edge_split:
        bpy.ops.object.modifier_add(type='EDGE_SPLIT')
        bpy.context.object.modifiers["EdgeSplit"].split_angle = 1.32645
        bpy.ops.object.modifier_apply(apply_as='DATA', modifier="EdgeSplit")

# Make light just directional, disable shadows.
lamp = bpy.data.lamps['Lamp']
lamp.type = 'SUN'
if args.remove_shadow:
    lamp.shadow_method = 'NOSHADOW'

# Possibly disable specular shading:
lamp.use_specular = False
lamp.energy = 1.0

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.lamp_add(type='SUN')
lamp2 = bpy.data.lamps['Sun']
lamp2.shadow_method = 'NOSHADOW'
lamp2.use_specular = False
lamp2.energy = 0.015
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Lamp'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def sample_spherical(radius=2, ndim=3):
    vec = np.random.randn(ndim)
    vec /= np.linalg.norm(vec, axis=0) 
    vec = vec * radius
    return tuple(vec)


scene = bpy.context.scene
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.alpha_mode = 'TRANSPARENT'
cam = scene.objects['Camera']

if args.camera_trace == 'fixed' or args.camera_trace == 'circle':
    assert args.views == 1 or args.camera_trace != 'fixed', 'fixed only support one position'
    cam.location = eval(args.camera_position)

elif args.camera_trace == 'sphere_random':
    cam.location = sample_spherical(args.camera_radius)

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

model_identifier = args.obj.split('/')[-3]

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

# make direc
os.makedirs(os.path.join(args.output_folder, model_identifier, 'extrinsic'))
with open(os.path.join(args.output_folder, model_identifier, 'intrinsic.txt'), 'w') as fk:
    print(matrix2str(get_calibration_matrix_K_from_blender(cam.data)), file=fk)

normal_file_output.base_path = ''

for i in range(0, args.views):

    if args.camera_trace == 'circle':
        stepsize = 360.0 / args.views
        rotation_mode = 'XYZ'
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
        
        filename = '_r_{0:03d}'.format(int(i * stepsize))
    else:
        filename = '_{0:03d}'.format(int(i))
    
    scene.render.filepath = os.path.join(args.output_folder, model_identifier, 'rgb', 'model' + filename)
    normal_file_output.file_slots[0].path = os.path.join(args.output_folder, model_identifier, 'normal', 'model' + filename + '.')

    with open(os.path.join(args.output_folder, model_identifier, 'extrinsic', 'model' + filename + '.txt'), 'w') as frt:
        print(matrix2str(get_3x4_RT_matrix_from_blender(cam)), file=frt)

    bpy.ops.render.render(write_still=True)  # render still

    if args.camera_trace == 'sphere_random':
        cam.location = sample_spherical(args.camera_radius)
    else:
        b_empty.rotation_euler[2] += radians(stepsize)