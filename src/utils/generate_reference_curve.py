import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt


def pick_points_from_plot():
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
    ax.plot(curve["x"], curve["y"], "r-")
    ax.plot(x_input, y_input, "bo")
    plt.draw()
    print("Press any key to exit")
    plt.waitforbuttonpress(0)
    plt.close(fig)
    return curve


def generate_reference_curve(xx, yy, delta):
    assert len(xx) == len(yy) >= 4
    delta_s = np.sqrt(np.diff(xx) ** 2 + np.diff(yy) ** 2)
    chords = np.cumsum(np.concatenate([[0], delta_s]))

    # Generate spline for xx(s) and yy(s)
    sp_x = interpolate.splrep(chords, xx)
    sp_y = interpolate.splrep(chords, yy)

    # At every delta meter, evaluate spline...
    s_sampled = np.arange(0, max(chords) + delta, delta)
    x_curve = interpolate.splev(s_sampled, sp_x, der=0)
    y_curve = interpolate.splev(s_sampled, sp_y, der=0)

    # ... and its first ...
    x_prime = interpolate.splev(s_sampled, sp_x, der=1)
    y_prime = interpolate.splev(s_sampled, sp_y, der=1)

    # ... and its second derivative ...
    x_pprime = interpolate.splev(s_sampled, sp_x, der=2)
    y_pprime = interpolate.splev(s_sampled, sp_y, der=2)

    # delta_s = sqrt(delta_x² + delta_y²) (add zero at the first
    s_curve = np.concatenate(
        (
            np.array([0]),
            np.cumsum(np.sqrt(np.diff(x_curve) ** 2 + np.diff(y_curve) ** 2)),
        )
    )

    # tan(theta) = dy/dx = (dy/ds) / (dx/ds)
    theta_curve = np.arctan2(y_prime, x_prime)

    # kappa = (x'y'' - y'x'')/(x'² + y'²)^(3/2)
    kappa_curve = (x_prime * y_pprime - y_prime * x_pprime) / (
        x_prime ** 2 + y_prime ** 2
    ) ** (3 / 2)

    return {
        "s": s_curve,
        "x": x_curve,
        "y": y_curve,
        "theta": theta_curve,
        "kappa": kappa_curve,
    }


def main():
    curve = pick_points_from_plot()
    print(curve)


if __name__ == "__main__":
    main()
