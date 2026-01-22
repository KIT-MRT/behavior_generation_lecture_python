"""State-based feedback controller for lateral vehicle control."""

import numpy as np

import behavior_generation_lecture_python.utils.normalize_angle as na


def feedback_law(
    lateral_error: float,
    vehicle_heading: float,
    reference_heading: float,
    reference_curvature: float,
) -> float:
    """Compute steering angle using state-based feedback control.

    This controller uses a linearized kinematic bicycle model and computes
    a steering command based on lateral error, heading error, and curvature
    feedforward.

    Args:
        lateral_error: Distance from vehicle to reference curve [m],
            positive if vehicle is left of the curve
        vehicle_heading: Heading angle of the vehicle [rad]
        reference_heading: Heading angle of the reference curve at the
            projection point [rad]
        reference_curvature: Curvature of the reference curve at the
            projection point [1/m]

    Returns:
        Steering angle command [rad]
    """
    wheelbase = 2.9680  # Distance between front and rear axle [m]
    lateral_error_gain = 0.2
    heading_error_gain = 1.0

    # Compute heading error with angle normalization
    heading_error = na.normalize_angle(vehicle_heading - reference_heading)

    # State feedback with curvature feedforward
    curvature_command = (
        reference_curvature
        - lateral_error_gain * lateral_error
        - heading_error_gain * heading_error
    )

    # Convert curvature to steering angle (inverse kinematic bicycle model)
    steering_angle = np.arctan(wheelbase * curvature_command)
    return steering_angle
