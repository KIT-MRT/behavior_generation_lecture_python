import numpy as np
import utils.normalize_angle as na


def feedback_law(d, psi, theta_r, kappa_r):
    """
    Feedback law for the state-based controller
    :param d: Distance of the vehicle to the reference curve
    :param psi: Heading of the vehicle
    :param theta_r: Heading of the reference line
    :param kappa_r: Curvature of the reference line
    :return: Steering angle
    """
    axis_distance = 2.9680
    k_0 = 0.2
    k_1 = 1.0

    # Stabilization
    u = kappa_r - k_0 * d - k_1 * na.normalize_angle(psi - theta_r)

    # Re-substitution
    delta = np.arctan(axis_distance * u)
    return delta
