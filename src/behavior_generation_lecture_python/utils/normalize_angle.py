"""Angle normalization utilities."""

import numpy as np


def normalize_angle(angle: float) -> float:
    """Normalize angle to be between -pi and pi.

    Args:
        angle: Angle in radians

    Returns:
        Normalized angle in the range [-pi, pi]
    """
    result: float = (angle + np.pi) % (2 * np.pi) - np.pi
    return result
