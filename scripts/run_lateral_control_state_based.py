import matplotlib.pyplot as plt
import numpy as np

import behavior_generation_lecture_python.lateral_control_state_based.lateral_control_state_based as cl
import behavior_generation_lecture_python.utils.generate_reference_curve as ref
from behavior_generation_lecture_python.lateral_control_state_based.lateral_control_state_based import (
    KinematicVehicleState,
)
from behavior_generation_lecture_python.utils.plot_vehicle import plot_vehicle as pv
from behavior_generation_lecture_python.utils.vizard import vizard as vz


def main() -> None:
    print("Running simulation...")
    radius = 20
    initial_state = KinematicVehicleState(
        x=0.1,
        y=float(-radius),
        heading=0.0,
    )
    curve = ref.generate_reference_curve(
        np.array([0, radius, 0, -radius, 0]),
        np.array([-radius, 0, radius, 0, radius]),
        1.0,
    )
    time_vector = np.arange(0, 100, 0.1)
    model = cl.LateralControlStateBased(initial_state, curve)
    trajectory = model.simulate(time_vector, velocity=1)

    # Extract data from ControllerOutput list
    x = np.array([out.x for out in trajectory])
    y = np.array([out.y for out in trajectory])
    psi = np.array([out.heading for out in trajectory])
    delta = np.array([out.steering_angle for out in trajectory])

    fig, ax = plt.subplots()

    plt.plot(curve.x, curve.y, "r-", linewidth=0.5)
    plt.plot(x, y, "b-", linewidth=0.5)
    plt.axis("equal")

    (point1,) = ax.plot([], [], marker="o", color="blue", ms=5)

    def update(i: int, *fargs: object) -> None:
        for line in reversed(ax.lines[1:]):
            line.remove()
        ax.plot(x[:i], y[:i], "b-", linewidth=0.5)
        point1.set_data(x[i : i + 1], y[i : i + 1])
        pv.plot_vehicle(ax, x[i], y[i], psi[i], delta[i])
        for farg in fargs:
            print(farg)

    vz.Vizard(figure=fig, update_func=update, time_vec=time_vector)
    plt.show()


if __name__ == "__main__":
    main()
