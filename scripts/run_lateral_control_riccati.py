import numpy as np
import matplotlib.pyplot as plt

import lateral_control_riccati.lateral_control_riccati as cl
import utils.generate_reference_curve as ref
from utils.plot_vehicle import plot_vehicle as pv
from utils.vizard import vizard as vz
from vehicle_models.vehicle_parameters import DEFAULT_VEHICLE_PARAMS


def main():
    print("Running simulation...")
    radius = 500
    vars_0 = [0.0, -radius, 0.0, 0.0, 0.0]
    v_0 = 33.0

    curve = ref.generate_reference_curve(
        [0, radius, 0, -radius, 0], [-radius, 0, radius, 0, radius], 10.0
    )
    ti = np.arange(0, 40, 0.1)

    # r = 10  # hectic steering behavior
    r = 10000  # fairly calm steering behavior

    model = cl.LateralControlRiccati(
        initial_condition=vars_0,
        curve=curve,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        initial_velocity=v_0,
        r=r,
    )
    sol = model.simulate(ti, v=v_0, t_step=0.1)
    x = sol[:, 0]
    y = sol[:, 1]
    psi = sol[:, 2]
    delta = sol[:, 5]

    fig, ax = plt.subplots()

    plt.plot(curve["x"], curve["y"], "r-", linewidth=0.5)
    plt.plot(x, y, "b-")
    plt.axis("equal")

    (point1,) = ax.plot([], [], marker="o", color="blue", ms=5)

    def update(i, *fargs):
        [l.remove() for l in reversed(ax.lines[1:])]
        ax.plot(x[: i + 1], y[: i + 1], "b-", linewidth=0.5)
        point1.set_data(x[i], y[i])
        pv.plot_vehicle(ax, x[i], y[i], psi[i], delta[i])
        for farg in fargs:
            print(farg)

    vz.Vizard(figure=fig, update_func=update, time_vec=ti)
    plt.show()


if __name__ == "__main__":
    main()
