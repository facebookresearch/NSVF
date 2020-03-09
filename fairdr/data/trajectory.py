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
def circle(radius=3.5, y=0.0):
    return lambda t: [radius * np.cos(t), radius * np.sin(t), y]
