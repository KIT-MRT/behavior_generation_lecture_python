"""LQR-based feedback controller for lateral vehicle control."""

from typing import Any

import numpy as np


def feedback_law(
    k_lqr: np.ndarray[Any, Any],
    k_dist_comp: float,
    lateral_error: float,
    heading_error: float,
    reference_curvature: float,
    beta: float,
    yaw_rate: float,
) -> float:
    """Compute steering angle using LQR feedback with disturbance compensation.

    The control law combines state feedback (LQR) with a feedforward term for
    curvature compensation. The state vector consists of lateral error, heading
    error, sideslip angle (beta), and yaw rate.

    Args:
        k_lqr: LQR gain vector [k_lateral, k_heading, k_beta, k_yaw_rate]
        k_dist_comp: Disturbance compensation gain for curvature feedforward
        lateral_error: Distance from vehicle to reference curve [m]
        heading_error: Difference between vehicle heading and reference heading [rad]
        reference_curvature: Curvature of the reference curve at the projection point [1/m]
        beta: Vehicle sideslip angle [rad]
        yaw_rate: Vehicle yaw rate [rad/s]

    Returns:
        Steering angle command [rad]
    """
    state = np.array([lateral_error, heading_error, beta, yaw_rate])
    steering_angle: float = float(np.dot(-k_lqr, state) + k_dist_comp * reference_curvature)

    return steering_angle
