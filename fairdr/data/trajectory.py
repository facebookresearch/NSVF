import numpy


def circle(radius=3.5, y=0.0):
    return lambda t: np.array([radius * np.cos(t), y, radius * np.sin(t)])