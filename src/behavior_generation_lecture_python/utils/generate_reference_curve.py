"""Generate reference curves from input points using spline interpolation."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

from behavior_generation_lecture_python.utils.reference_curve import ReferenceCurve


def pick_points_from_plot() -> ReferenceCurve:
    """Interactively pick points from a plot to generate a reference curve.

    Returns:
        A ReferenceCurve generated from the selected points.
    """
    fig, ax = plt.subplots()
    ax.set_xlim([0, 100])
    ax.set_ylim([0, 100])
    ax.set_aspect("equal", "box")
    while True:
        print(
            "Please select 4 or more points for the spline! Press Enter when you're done."
        )
        xy = np.array(
            plt.ginput(n=0, timeout=0)
        )  # infinite number of points, not timeout
        if len(xy) > 3:  # Spline requires more than 3 points.
            break
    x_input = xy[:, 0]
    y_input = xy[:, 1]
    curve = generate_reference_curve(x_input, y_input, 1)
    ax.plot(curve.x, curve.y, "r-")
    ax.plot(x_input, y_input, "bo")
    plt.draw()
    print("Press any key to exit")
    plt.waitforbuttonpress(0)
    plt.close(fig)
    return curve


def generate_reference_curve(
    x_points: np.ndarray[Any, Any], y_points: np.ndarray[Any, Any], sampling_distance: float
) -> ReferenceCurve:
    """Generate a reference curve from input points using spline interpolation.

    Args:
        x_points: X-coordinates of the input points (at least 4 points required)
        y_points: Y-coordinates of the input points (at least 4 points required)
        sampling_distance: Distance between sampled points on the output curve [m]

    Returns:
        A ReferenceCurve with arc length, x, y, heading, and curvature arrays.
    """
    assert len(x_points) == len(y_points) >= 4
    segment_lengths = np.sqrt(np.diff(x_points) ** 2 + np.diff(y_points) ** 2)
    chord_lengths = np.cumsum(np.concatenate([[0], segment_lengths]))

    # Generate spline for x(s) and y(s)
    spline_x = interpolate.splrep(chord_lengths, x_points)
    spline_y = interpolate.splrep(chord_lengths, y_points)

    # At every sampling_distance meter, evaluate spline...
    arc_length_sampled = np.arange(
        0, max(chord_lengths) + sampling_distance, sampling_distance
    )
    x_curve = interpolate.splev(arc_length_sampled, spline_x, der=0)
    y_curve = interpolate.splev(arc_length_sampled, spline_y, der=0)

    # ... and its first derivative ...
    dx_ds = interpolate.splev(arc_length_sampled, spline_x, der=1)
    dy_ds = interpolate.splev(arc_length_sampled, spline_y, der=1)

    # ... and its second derivative ...
    d2x_ds2 = interpolate.splev(arc_length_sampled, spline_x, der=2)
    d2y_ds2 = interpolate.splev(arc_length_sampled, spline_y, der=2)

    # Compute arc length: delta_s = sqrt(delta_x^2 + delta_y^2)
    arc_length = np.concatenate(
        (
            np.array([0]),
            np.cumsum(np.sqrt(np.diff(x_curve) ** 2 + np.diff(y_curve) ** 2)),
        )
    )

    # Heading: tan(theta) = dy/dx = (dy/ds) / (dx/ds)
    heading = np.arctan2(dy_ds, dx_ds)

    # Curvature: kappa = (x'y'' - y'x'') / (x'^2 + y'^2)^(3/2)
    curvature = (dx_ds * d2y_ds2 - dy_ds * d2x_ds2) / (dx_ds**2 + dy_ds**2) ** (3 / 2)

    return ReferenceCurve(
        arc_length=arc_length,
        x=x_curve,
        y=y_curve,
        heading=heading,
        curvature=curvature,
    )


def main() -> None:
    curve = pick_points_from_plot()
    print(curve)


if __name__ == "__main__":
    main()
