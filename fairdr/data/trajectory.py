import torch
import numpy as np

TRAJECTORY_REGISTRY = {}


def register_traj(name):
    def register_traj_fn(fn):
        if name in TRAJECTORY_REGISTRY:
            raise ValueError('Cannot register duplicate trajectory ({})'.format(name))
        TRAJECTORY_REGISTRY[name] = fn
        return fn
    return register_traj_fn


def get_trajectory(name):
    return TRAJECTORY_REGISTRY.get(name, None)


@register_traj('circle')
def circle(radius=3.5, h=0.0, axis='z'):
    if axis == 'z':
        return lambda t: [radius * np.cos(t), radius * np.sin(t), h]
    elif axis == 'y':
        return lambda t: [radius * np.cos(t), h, radius * np.sin(t)]
    else:
        return lambda t: [h, radius * np.cos(t), radius * np.sin(t)]