import numpy as np


def normalize_angle(angle):
    """Normalize angle to be between -pi and pi"""
    return (angle + np.pi) % (2 * np.pi) - np.pi
