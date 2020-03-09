import importlib

bpy_loader= importlib.find_loader('bpy')
has_bpy= bpy_loader is not None
if has_bpy:
    import bpy
    import bpy_extras
    import mathutils
    from mathutils import Vector
    from mathutils import Matrix

import numpy as np
import os
import shutil
def calc_ani_extent():
    mn = None
    mx = None
    for i in range(1081):
        bpy.context.scene.frame_set(i)
        bpy.context.scene.update()
        obj = bpy.data.objects['1876.001']
        bb_vertices = [Vector(v) for v in obj.bound_box]
        mat = obj.matrix_world

        world_bb_vertices = [mat * v for v in bb_vertices]
        bbox = np.array(world_bb_vertices)

        if mn is None:
            mn = np.amin(bbox,axis=0)
            mx = np.amax(bbox,axis=0)
        mnc = np.amin(bbox,axis=0)
        mxc = np.amax(bbox,axis=0)
        mn = np.amin(np.vstack((mn,mnc)),axis=0)
        mx = np.amax(np.vstack((mn,mxc)),axis=0)

    bpy.data.objects['AniExtent'].data.vertices[0].co = (mn[0],mn[1],mn[2])
    bpy.data.objects['AniExtent'].data.vertices[1].co = (mn[0],mx[1],mn[2])
    bpy.data.objects['AniExtent'].data.vertices[2].co = (mn[0],mn[1],mx[2])
    bpy.data.objects['AniExtent'].data.vertices[3].co = (mn[0],mx[1],mx[2])
    bpy.data.objects['AniExtent'].data.vertices[4].co = (mx[0],mn[1],mn[2])
    bpy.data.objects['AniExtent'].data.vertices[5].co = (mx[0],mx[1],mn[2])
    bpy.data.objects['AniExtent'].data.vertices[6].co = (mx[0],mn[1],mx[2])
    bpy.data.objects['AniExtent'].data.vertices[7].co = (mx[0],mx[1],mx[2])



#---------------------------------------------------------------
# 3x4 P matrix from Blender camera
#---------------------------------------------------------------

# BKE_camera_sensor_size
def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

# BKE_camera_sensor_fit
def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit

# Build intrinsic camera parameters from Blender camera data
#
# See notes on this in
# blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# as well as
# https://blender.stackexchange.com/a/120063/3581
def get_calibration_matrix_K_from_blender(camd):
    if camd.type != 'PERSP':
        raise ValueError('Non-perspective cameras not supported')
    scene = bpy.context.scene
    f_in_mm = camd.lens
    scale = scene.render.resolution_percentage / 100
    resolution_x_in_px = scale * scene.render.resolution_x
    resolution_y_in_px = scale * scene.render.resolution_y
    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
    sensor_fit = get_sensor_fit(
        camd.sensor_fit,
        scene.render.pixel_aspect_x * resolution_x_in_px,
        scene.render.pixel_aspect_y * resolution_y_in_px
    )
    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
    if sensor_fit == 'HORIZONTAL':
        view_fac_in_px = resolution_x_in_px
    else:
        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
    s_u = 1 / pixel_size_mm_per_px
    s_v = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

    # Parameters of intrinsic calibration matrix K
    u_0 = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
    v_0 = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio
    skew = 0 # only use rectangular pixels

    K = Matrix(
        ((s_u, skew, u_0),
        (   0,  s_v, v_0),
        (   0,    0,   1)))
    return K

# Returns camera rotation and translation matrices from Blender.
#
# There are 3 coordinate systems involved:
#    1. The World coordinates: "world"
#       - right-handed
#    2. The Blender camera coordinates: "bcam"
#       - x is horizontal
#       - y is up
#       - right-handed: negative z look-at direction
#    3. The desired computer vision camera coordinates: "cv"
#       - x is horizontal
#       - y is down (to align to the actual pixel coordinates
#         used in digital images)
#       - right-handed: positive z look-at direction
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    R_bcam2cv = Matrix(
        ((1, 0,  0),
        (0, -1, 0),
        (0, 0, -1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam * location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam*cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1*R_world2bcam * location

    # Build the coordinate transform matrix from world to computer vision camera
    R_world2cv = R_bcam2cv*R_world2bcam
    T_world2cv = R_bcam2cv*T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2cv[0][:] + (T_world2cv[0],),
        R_world2cv[1][:] + (T_world2cv[1],),
        R_world2cv[2][:] + (T_world2cv[2],)
        ))
    return RT

def get_3x4_P_matrix_from_blender(cam):
    K = get_calibration_matrix_K_from_blender(cam.data)
    RT = get_3x4_RT_matrix_from_blender(cam)
    return K*RT, K, RT

def b2dv(cam_id,pose_path):

    cam = bpy.data.objects[cam_id]
    P, K, RT = get_3x4_P_matrix_from_blender(cam)
    RT=np.vstack((RT,np.array([0,0,0,1])))
    RT=np.linalg.inv(RT)#make it cam2world
    np.savetxt(pose_path, RT,fmt='%0.9f')

    #f  = open(pose_path+'.frame', 'w')
    #f.write('%d'%bpy.context.scene.frame_current)
    #f.close()


import numpy as np

def archimedean_spiral(points=500., origin=np.array([0., 0., 0.])):
    a = 300
    r = 1#sphere_radius
    o = origin

    translations = []

    i = a / 2
    while i > 0.:
        x = r * np.cos(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        y = r * np.sin(i) * np.cos((-np.pi / 2) + i / a * np.pi)
        z = r * - np.sin(-np.pi / 2 + i / a * np.pi)
        xyz = np.array((x,y,z)) + o
        translations.append(xyz)
        i -= a / (2 * points)

    return np.array(translations)

def half_sphere_golden_spiral(num_pts=1000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(phi)
    idx = z>0
    x=x[idx]
    y=y[idx]
    z=z[idx]
    points = np.stack((x,y,z),axis=-1)
    return points


def parse_cameras(path):
    import glob
    import numpy as np
    from pathlib import Path

    poses = glob.glob(path+'/*.txt')
    cameras = {}
    for ppath in poses:

        cam_name = Path(ppath).stem
        RT = np.loadtxt(ppath)
        cameras[cam_name]=RT
    return cameras

def save_data(frame_id,cam_id,root = '.',scene_id='blender_render' ):
    scene = bpy.context.scene
    root = '%s/%s/%04d/'%(root,scene_id,frame_id)
    rgbdir = root+'/rgb/'
    posedir = root+'/pose/'
    depthdir = root+'/depth/'
    if not os.path.exists(rgbdir):os.makedirs(rgbdir)
    if not os.path.exists(posedir):os.makedirs(posedir)
    if not os.path.exists(depthdir):os.makedirs(depthdir)
    #scene.render.filepath = root+'/rgb/'+did+'_'+id
    bpy.data.scenes[0].node_tree.nodes['rgb_output'].base_path = rgbdir
    bpy.data.scenes[0].node_tree.nodes['depth_output'].base_path = depthdir

    bpy.data.scenes[0].node_tree.nodes['rgb_output'].file_slots[0].path = cam_id+'.png'
    bpy.data.scenes[0].node_tree.nodes['depth_output'].file_slots[0].path = cam_id+'.exr'

    posedir += cam_id+'.txt'
    b2dv(cam_id,posedir)
    bpy.ops.render.render(write_still=True)
    shutil.move(rgbdir+cam_id+'.png%04d'%frame_id,rgbdir+cam_id+'.png')
    shutil.move(depthdir+cam_id+'.exr%04d'%frame_id,depthdir+cam_id+'.exr')

def render(frame_id,cam_id,save_path='.',scene_id='render'):
    bpy.context.scene.frame_set(frame_id)
    bpy.context.scene.camera = bpy.data.objects[cam_id]
    save_data(frame_id,cam_id,save_path,scene_id)

def cams_sphere(num_pts):
    #points = lib_math.archimedean_spiral(num_pts) *3.5
    points = half_sphere_golden_spiral(num_pts) *3.5
    traj_id = 'cam_train'
    group = bpy.data.groups.new(traj_id)
    for i, point in enumerate(points):
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name='%s_%04d'%(traj_id,i)

        cam.location = mathutils.Vector(point)
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.location.to_track_quat('Z', 'Y')

        group.objects.link(cam)

def create_cameras(num_pts,cam_name='cam_train'):
    #points = lib_math.archimedean_spiral(num_pts) *3.5
    points = half_sphere_golden_spiral(num_pts) *3.5
    traj_id = cam_name #'cam_train'
    group = bpy.data.groups.new(traj_id)
    for i, point in enumerate(points):
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.name='%s_%04d'%(traj_id,i)

        cam.location = mathutils.Vector(point)
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = cam.location.to_track_quat('Z', 'Y')
        group.objects.link(cam)
