# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys, os, argparse
import json
import bpy
import mathutils
from mathutils import Vector
import numpy as np

np.random.seed(2)  # fixed seed

DEBUG = False
VOXEL_NUMS = 512
VIEWS = 200
RESOLUTION = 800
RESULTS_PATH = 'rgb'
DEPTH_SCALE = 1.4
COLOR_DEPTH = 8
FORMAT = 'PNG'
RANDOM_VIEWS = True
UPPER_VIEWS = True
CIRCLE_FIXED_START = (.3,0,0)

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('output', type=str, help='path where files will be saved')

argv = sys.argv
argv = argv[argv.index("--") + 1:]
args = parser.parse_args(argv)

homedir = args.output
fp = bpy.path.abspath(f"{homedir}/{RESULTS_PATH}")

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

if not os.path.exists(fp):
    os.makedirs(fp)
if not os.path.exists(os.path.join(homedir, "pose")):
    os.mkdir(os.path.join(homedir, "pose"))

# Data to store in JSON file
out_data = {
    'camera_angle_x': bpy.data.objects['Camera'].data.angle_x,
}

# Render Optimizations
bpy.context.scene.render.use_persistent_data = True


# Set up rendering of depth map.
bpy.context.scene.use_nodes = True
tree = bpy.context.scene.node_tree
links = tree.links

# Add passes for additionally dumping albedo and normals.
#bpy.context.scene.view_layers["RenderLayer"].use_pass_normal = True
bpy.context.scene.render.image_settings.file_format = str(FORMAT)
bpy.context.scene.render.image_settings.color_depth = str(COLOR_DEPTH)

if not DEBUG:
    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    if FORMAT == 'OPEN_EXR':
      links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
    else:
      # Remap as other types can not represent the full range of depth.
      map = tree.nodes.new(type="CompositorNodeMapValue")
      # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
      map.offset = [-0.7]
      map.size = [DEPTH_SCALE]
      map.use_min = True
      map.min = [0]
      links.new(render_layers.outputs['Depth'], map.inputs[0])

      links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

# Background
bpy.context.scene.render.dither_intensity = 0.0
bpy.context.scene.render.film_transparent = True

# Create collection for objects not to render with background
objs = [ob for ob in bpy.context.scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
bpy.ops.object.delete({"selected_objects": objs})

# bounding box
for obj in bpy.context.scene.objects:
    if 'Camera' not in obj.name:
        bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        bbox = [min([bb[i] for bb in bbox]) for i in range(3)] + \
               [max([bb[i] for bb in bbox]) for i in range(3)]
        voxel_size = ((bbox[3]-bbox[0]) * (bbox[4]-bbox[1]) * (bbox[5]-bbox[2]) / VOXEL_NUMS) ** (1/3)
        print(" ".join(['{:.5f}'.format(f) for f in bbox + [voxel_size]]), 
            file=open(os.path.join(homedir, 'bbox.txt'), 'w'))

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    # scn.objects.active = b_empty
    return b_empty


scene = bpy.context.scene
scene.render.resolution_x = RESOLUTION
scene.render.resolution_y = RESOLUTION
scene.render.resolution_percentage = 100

cam = scene.objects['Camera']
cam.location = (4, -4, 4)
cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'
b_empty = parent_obj_to_camera(cam)
cam_constraint.target = b_empty

scene.render.image_settings.file_format = 'PNG'  # set output format to .png

from math import radians

stepsize = 360.0 / VIEWS
rotation_mode = 'XYZ'

if not DEBUG:
    for output_node in [depth_file_output, normal_file_output]:
        output_node.base_path = ''

out_data['frames'] = []

if not RANDOM_VIEWS:
    b_empty.rotation_euler = CIRCLE_FIXED_START

for i in range(0, VIEWS):
    if DEBUG:
        i = np.random.randint(0,VIEWS)
        b_empty.rotation_euler[2] += radians(stepsize*i)
    if RANDOM_VIEWS:
        scene.render.filepath = os.path.join(fp, '{:04d}'.format(i))
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
        scene.render.filepath = os.path.join(fp, '{:04d}'.format(i))

    # depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
    # normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"
    print('BEFORE RENDER')
    if DEBUG:
        break
    else:
        bpy.ops.render.render(write_still=True)  # render still
    print('AFTER RENDER')

    frame_data = {
        'file_path': scene.render.filepath,
        'rotation': radians(stepsize),
        'transform_matrix': listify_matrix(cam.matrix_world)
    }
    with open(os.path.join(homedir, "pose", '{:04d}.txt'.format(i)), 'w') as fo:
        for ii, pose in enumerate(frame_data['transform_matrix']):
            print(" ".join([str(-p) if (((j == 2) | (j == 1)) and (ii < 3)) else str(p) 
                            for j, p in enumerate(pose)]), 
                file=fo)
    out_data['frames'].append(frame_data)

    if RANDOM_VIEWS:
        if UPPER_VIEWS:
            rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
            rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
            b_empty.rotation_euler = rot
        else:
            b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
    else:
        b_empty.rotation_euler[2] += radians(stepsize)

if not DEBUG:
    with open(os.path.join(homedir, 'transforms.json'), 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


# save camera data
H, W = RESOLUTION, RESOLUTION
f = .5 * W /np.tan(.5 * float(out_data['camera_angle_x']))
cx = cy = W // 2

# write intrinsics
with open(os.path.join(homedir, 'intrinsics.txt'), 'w') as fi:
    print("{} {} {} 0.".format(f, cx, cy), file=fi)
    print("0. 0. 0.", file=fi)
    print("0.", file=fi)
    print("1.", file=fi)
    print("{} {}".format(H, W), file=fi)