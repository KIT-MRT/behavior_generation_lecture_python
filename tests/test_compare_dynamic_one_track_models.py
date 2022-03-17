import numpy as np
import pytest

import behavior_generation_lecture_python.vehicle_models.model_comparison as cm


@pytest.mark.parametrize(
    "steering_wheel_angle_amplitude,beta_error,r_error",
    [(20, 0.0001, 0.0003), (100, 0.8, 0.8)],
)
def test_model_comparison(steering_wheel_angle_amplitude, beta_error, r_error):
    def delta(t):
        tp = [0.0, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1]
        dp = [0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0]
        stwa = np.interp(t, tp, dp)

        stwa_ampl = steering_wheel_angle_amplitude * np.pi / 180
        stwa_ratio = 18

        stwa_max = 520 * np.pi / 180
        stwa = max(min(stwa_max, stwa), -stwa_max)

        delta = stwa_ampl * stwa / stwa_ratio

        return delta

    initial_condition = [0.0, 0.0, 0.0, 0.0, 0.0]
    timesteps = np.arange(0, 5, 0.05)
    model = cm.CompareModels(initial_condition=initial_condition, delta_fun=delta)
    sol = model.simulate(t_vector=timesteps, v=30)
    beta = sol[0][:, 3]
    r = sol[0][:, 4]
    beta_lin = sol[1][:, 0]
    r_lin = sol[1][:, 1]

    abs_diff_r = np.absolute(r - r_lin)
    abs_diff_beta = np.absolute(beta - beta_lin)

    assert np.mean(abs_diff_beta) < beta_error
    assert np.mean(abs_diff_r) < r_error
