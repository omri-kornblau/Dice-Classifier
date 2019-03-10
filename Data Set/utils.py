import numpy as np

def to_polar(a):
    amplitude = np.linalg.norm(a)
    theta = np.arctan2(a[2], a[1])
    gamma = np.arctan2(a[1], a[0])

    return np.array([amplitude, theta, gamma])

