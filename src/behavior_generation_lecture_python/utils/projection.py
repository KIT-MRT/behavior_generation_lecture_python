import warnings

import numpy as np
from scipy import spatial

import behavior_generation_lecture_python.utils.normalize_angle as na


def project2curve_with_lookahead(s_c, x_c, y_c, theta_c, kappa_c, l_v, x, y, psi):
    # Calculate look-ahead sensor point
    x = x + l_v * np.cos(psi)
    y = y + l_v * np.sin(psi)

    projection = project2curve(
        s_c=s_c, x_c=x_c, y_c=y_c, theta_c=theta_c, kappa_c=kappa_c, x=x, y=y
    )

    # Simulate camera view
    projection[4] = psi - projection[4]

    return projection


def project2curve(s_c, x_c, y_c, theta_c, kappa_c, x, y):
    """Project a point onto a curve (defined as a polygonal chain/ sequence of points/ line string)

    Args:
        s_c: Arc lenght of the curve
        x_c: x-coordinates of the curve points
        y_c: y-coordinates of the curve points
        theta_c: heading at the curve points
        kappa_c: curvature at the curve points
        x: x-coordinates of the point to be projected
        y: y-coordinates of the point to be projected

    Returns:
        properties of the projected point as list: x-coordinate,
        y-coordinate, arc length, distance to original point, heading,
        curvature
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

    return [x_p, y_p, s_p, d, theta_p, kappa_p]


def pseudo_projection(start_index, x, y, x_c, y_c, theta_c):
    """Project a point onto a segment of a curve (defined as a polygonal chain/ sequence of points/ line string)

    Args:
        start_index: Start index of the segment to be projected to
        x_c: x-coordinates of the curve points
        y_c: y-coordinates of the curve points
        theta_c: heading at the curve points
        x: x-coordinates of the point to be projected
        y: y-coordinates of the point to be projected

    Returns:
        properties of the projected point as list: point ([x-coordinate,
        y-coordinate]), lambda: interpolation scale, sgn: sign of the
        projection (-1 or 1)
    """
    p1 = np.array([x_c[start_index], y_c[start_index]])
    p2 = np.array([x_c[start_index + 1], y_c[start_index + 1]])
    theta1 = theta_c[start_index]
    theta1 = na.normalize_angle(theta1)
    theta2 = theta_c[start_index + 1]
    theta2 = na.normalize_angle(theta2)
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

    return [px, lambda_, sgn]
