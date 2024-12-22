import matplotlib.pyplot as plt
import numpy as np

import behavior_generation_lecture_python.vehicle_models.model_comparison as cm
from behavior_generation_lecture_python.utils.plot_vehicle import plot_vehicle as pv
from behavior_generation_lecture_python.utils.vizard import vizard as vz


def main():
    print("Running simulation...")

    def delta(t):
        stwa = 0
        tp = [0.0, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
        dp = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0]
        stwa = np.interp(t, tp, dp)

        stwa_ampl = 20 * np.pi / 180
        stwa_ratio = 18

        stwa_max = 520 * np.pi / 180
        stwa = max(min(stwa_max, stwa), -stwa_max)

        delta = stwa_ampl * stwa / stwa_ratio

        return delta

    initial_condition = [0.0, 0.0, 0.0, 0.0, 0.0]
    timesteps = np.arange(0, 5, 0.05)
    model = cm.CompareModels(initial_condition=initial_condition, delta_fun=delta)
    sol = model.simulate(t_vector=timesteps, v=30)
    x = sol[0][:, 0]
    y = sol[0][:, 1]
    psi = sol[0][:, 2]
    beta = sol[0][:, 3]
    r = sol[0][:, 4]
    beta_lin = sol[1][:, 0]
    r_lin = sol[1][:, 1]

    delta_vals = [delta(t) for t in timesteps]
    fig, (ax1, ax2) = plt.subplots(2)

    ax1.axis("equal")

    ax2.plot(timesteps, delta_vals, "k-")
    ax2.plot(timesteps, beta, "r-")
    ax2.plot(timesteps, r, "g-")
    ax2.plot(timesteps, beta_lin, "m-")
    ax2.plot(timesteps, r_lin, "b-")

    ax2.legend(["delta", "beta", "r", "beta_lin", "r_lin"])

    (point1,) = ax1.plot([], [], marker="o", color="blue", ms=10)
    (point_delta,) = ax2.plot([], [], marker="o", color="black", ms=3)
    (point_beta,) = ax2.plot([], [], marker="o", color="red", ms=3)
    (point_r,) = ax2.plot([], [], marker="o", color="green", ms=3)
    (point_beta_lin,) = ax2.plot([], [], marker="o", color="magenta", ms=3)
    (point_r_lin,) = ax2.plot([], [], marker="o", color="blue", ms=3)

    def update(i, *fargs):
        slice_ = slice(i + 1, i + 2)
        [l.remove() for l in reversed(ax1.lines)]
        ax1.plot(x[: i + 1], y[: i + 1], "b-", linewidth=0.5)
        point1.set_data(x[slice_], y[slice_])
        pv.plot_vehicle(ax1, x[i], y[i], psi[i], delta_vals[i])

        point_delta.set_data(timesteps[slice_], delta_vals[slice_])
        point_beta.set_data(timesteps[slice_], beta[slice_])
        point_r.set_data(timesteps[slice_], r[slice_])
        point_beta_lin.set_data(timesteps[slice_], beta_lin[slice_])
        point_r_lin.set_data(timesteps[slice_], r_lin[slice_])
        for farg in fargs:
            print(farg)

    vz.Vizard(figure=fig, update_func=update, time_vec=timesteps)
    plt.show()


if __name__ == "__main__":
    main()
