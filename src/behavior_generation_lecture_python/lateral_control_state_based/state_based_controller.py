import numpy as np

import behavior_generation_lecture_python.utils.normalize_angle as na


def feedback_law(d, psi, theta_r, kappa_r):
    """Feedback law for the state-based controller

    Args:
        d: Distance of the vehicle to the reference curve
        psi: Heading of the vehicle
        theta_r: Heading of the reference line
        kappa_r: Curvature of the reference line

    Returns:
        Steering angle
    """
    axis_distance = 2.9680
    k_0 = 0.2
    k_1 = 1.0

    # Stabilization
    u = kappa_r - k_0 * d - k_1 * na.normalize_angle(psi - theta_r)

    # Re-substitution
    delta = np.arctan(axis_distance * u)
    return delta
