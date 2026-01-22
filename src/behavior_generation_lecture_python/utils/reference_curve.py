"""Reference curve dataclass for path following controllers."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceCurve:
    """A reference curve for path following, defined by sampled points.

    Attributes:
        arc_length: Arc length values along the curve [m]
        x: X-coordinates of curve points [m]
        y: Y-coordinates of curve points [m]
        heading: Heading angle at each point [rad]
        curvature: Curvature at each point [1/m]
    """

    arc_length: np.ndarray
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    curvature: np.ndarray
