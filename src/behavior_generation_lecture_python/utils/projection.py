"""Projection utilities for projecting points onto reference curves."""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import spatial

import behavior_generation_lecture_python.utils.normalize_angle as na


@dataclass
class CurveProjection:
    """Result of projecting a point onto a reference curve.

    Attributes:
        x: X-coordinate of the projected point [m]
        y: Y-coordinate of the projected point [m]
        arc_length: Arc length along the curve at the projection point [m]
        lateral_error: Signed distance from original point to curve [m],
            positive if point is left of the curve
        heading: Heading angle at the projection point [rad]
        curvature: Curvature at the projection point [1/m]
    """

    x: float
    y: float
    arc_length: float
    lateral_error: float
    heading: float
    curvature: float


def project2curve_with_lookahead(
    s_c: np.ndarray[Any, Any],
    x_c: np.ndarray[Any, Any],
    y_c: np.ndarray[Any, Any],
    theta_c: np.ndarray[Any, Any],
    kappa_c: np.ndarray[Any, Any],
    lookahead_distance: float,
    x: float,
    y: float,
    psi: float,
) -> CurveProjection:
    """Project a point with look-ahead onto a reference curve.

    Computes the look-ahead sensor point and projects it onto the curve.
    The heading in the result is the heading error (vehicle heading - reference heading).

    Args:
        s_c: Arc length of the curve points
        x_c: X-coordinates of the curve points
        y_c: Y-coordinates of the curve points
        theta_c: Heading at the curve points
        kappa_c: Curvature at the curve points
        lookahead_distance: Look-ahead distance from vehicle position [m]
        x: X-coordinate of the vehicle
        y: Y-coordinate of the vehicle
        psi: Heading of the vehicle [rad]

    Returns:
        CurveProjection with heading representing heading error (psi - reference_heading)
    """
    # Calculate look-ahead sensor point
    x_lookahead = x + lookahead_distance * np.cos(psi)
    y_lookahead = y + lookahead_distance * np.sin(psi)

    projection = project2curve(
        s_c=s_c,
        x_c=x_c,
        y_c=y_c,
        theta_c=theta_c,
        kappa_c=kappa_c,
        x=x_lookahead,
        y=y_lookahead,
    )

    # Simulate camera view: return heading error instead of reference heading
    heading_error = psi - projection.heading

    return CurveProjection(
        x=projection.x,
        y=projection.y,
        arc_length=projection.arc_length,
        lateral_error=projection.lateral_error,
        heading=heading_error,
        curvature=projection.curvature,
    )


def project2curve(
    s_c: np.ndarray[Any, Any],
    x_c: np.ndarray[Any, Any],
    y_c: np.ndarray[Any, Any],
    theta_c: np.ndarray[Any, Any],
    kappa_c: np.ndarray[Any, Any],
    x: float,
    y: float,
) -> CurveProjection:
    """Project a point onto a curve (defined as a polygonal chain).

    Args:
        s_c: Arc length of the curve points
        x_c: X-coordinates of the curve points
        y_c: Y-coordinates of the curve points
        theta_c: Heading at the curve points
        kappa_c: Curvature at the curve points
        x: X-coordinate of the point to be projected
        y: Y-coordinate of the point to be projected

    Returns:
        CurveProjection containing projected point properties
    """
    # Find the closest curve point to [x, y]
    distance, mindex = spatial.KDTree(np.array([x_c, y_c]).transpose()).query([x, y])

    if mindex == 0:  # at the beginning
        start_index = 0
        px_, lambda_, sign = pseudo_projection(start_index, x, y, x_c, y_c, theta_c)
        if lambda_ < 0:
            warnings.warn("Extrapolating over start of reference curve!")
    elif mindex == len(s_c) - 1:  # at the end
        start_index = mindex - 1
        px_, lambda_, sign = pseudo_projection(start_index, x, y, x_c, y_c, theta_c)
        if lambda_ > 1:
            warnings.warn("Extrapolating over end of reference curve!")
    else:  # in between
        start_index = mindex
        px_, lambda_, sign = pseudo_projection(start_index, x, y, x_c, y_c, theta_c)
        if (
            lambda_ < 0
        ):  # Special case of variable distance sampling might require to shift start_index up.
            start_index = mindex - 1
            px_, lambda_, sign = pseudo_projection(start_index, x, y, x_c, y_c, theta_c)

        assert 0.0 <= lambda_ <= 1.0

    x_p = px_[0]
    y_p = px_[1]
    s1 = s_c[start_index]
    s2 = s_c[start_index + 1]
    s_p = lambda_ * s2 + (1.0 - lambda_) * s1

    d = sign * np.sqrt((x_p - x) ** 2 + (y_p - y) ** 2)

    theta1 = theta_c[start_index]
    theta2 = theta_c[start_index + 1]
    delta_theta = theta2 - theta1
    delta_theta = na.normalize_angle(delta_theta)
    theta_p = theta1 + lambda_ * delta_theta
    theta_p = na.normalize_angle(theta_p)

    kappa1 = kappa_c[start_index]
    kappa2 = kappa_c[start_index + 1]
    kappa_p = lambda_ * kappa2 + (1.0 - lambda_) * kappa1

    return CurveProjection(
        x=x_p,
        y=y_p,
        arc_length=s_p,
        lateral_error=d,
        heading=theta_p,
        curvature=kappa_p,
    )


def pseudo_projection(
    start_index: int,
    x: float,
    y: float,
    x_c: np.ndarray[Any, Any],
    y_c: np.ndarray[Any, Any],
    theta_c: np.ndarray[Any, Any],
) -> tuple[np.ndarray[Any, Any], float, int]:
    """Project a point onto a segment of a curve.

    Args:
        start_index: Start index of the segment to be projected to
        x: X-coordinate of the point to be projected
        y: Y-coordinate of the point to be projected
        x_c: X-coordinates of the curve points
        y_c: Y-coordinates of the curve points
        theta_c: Heading at the curve points

    Returns:
        Tuple of (projected_point, lambda, sign) where:
        - projected_point: [x, y] coordinates of projection
        - lambda: interpolation parameter (0 to 1)
        - sign: +1 if point is left of curve, -1 if right, 0 if on curve
    """
    p1 = np.array([x_c[start_index], y_c[start_index]])
    p2 = np.array([x_c[start_index + 1], y_c[start_index + 1]])
    theta1 = theta_c[start_index]
    theta2 = theta_c[start_index + 1]
    delta = np.array(p2) - np.array(p1)
    length = np.linalg.norm(delta)

    # transform so that origin is p1 oriented to p2
    alpha = np.arctan2(delta[1], delta[0])
    sin_ = np.sin(-alpha)
    cos_ = np.cos(-alpha)
    R = np.array([[cos_, -sin_], [sin_, cos_]])
    x_ = np.dot(R, (np.array([x, y]) - np.array(p1)))
    m1 = np.tan(theta1 - alpha)
    m2 = np.tan(theta2 - alpha)
    devi = (m1 - m2) * x_[1] + length
    if abs(devi) > 0.001:
        lambda_ = (m1 * x_[1] + x_[0]) / devi
    else:
        lambda_ = 0.5

    px = lambda_ * np.array(p2) + (1.0 - lambda_) * np.array(p1)

    sgn = 0
    if x_[1] > 0:
        sgn = 1
    elif x_[1] < 0:
        sgn = -1

    return (px, float(lambda_), sgn)
