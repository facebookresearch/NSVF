import torch
import numpy as np

def circle(radius=3.5, y=0.0):
    return lambda t: [radius * np.cos(t), y, radius * np.sin(t)]