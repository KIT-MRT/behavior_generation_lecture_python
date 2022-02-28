import numpy as np
from scipy.integrate import odeint

from vehicle_models.vehicle_parameters import VehicleParameters, DEFAULT_VEHICLE_PARAMS


class DynamicOneTrackModel:
    def __init__(self, params: VehicleParameters = None):
        if params is None:
            self.params = DEFAULT_VEHICLE_PARAMS
        else:
            self.params = params

    def system_results(self, vars_, v, delta):
        x, y, psi, beta, r = vars_

        x_dot = v * np.cos(psi + beta)
        y_dot = v * np.sin(psi + beta)
        psi_dot = r

        alpha_v = delta - np.arctan(
            (self.params.l_v * r + v * np.sin(beta)) / (v * np.cos(beta))
        )
        alpha_h = np.arctan(
            (self.params.l_h * r - v * np.sin(beta)) / (v * np.cos(beta))
        )

        s_qv = np.arctan(alpha_v)
        s_qh = np.arctan(alpha_h)

        F_qv = self.params.A_v * np.sin(
            self.params.B_v * np.arctan(self.params.C_v * s_qv)
        )
        F_qh = self.params.A_h * np.sin(
            self.params.B_h * np.arctan(self.params.C_h * s_qh)
        )

        r_dot = (
            -self.params.l_h * F_qh + F_qv * self.params.l_v * np.cos(delta)
        ) / self.params.J
        beta_dot = -r + (F_qv * np.cos(delta - beta) + F_qh * np.cos(beta)) / (
            self.params.m * v
        )

        v_dot = 0  # v = const
        a_x = v_dot * np.cos(beta) - v * (r + beta_dot) * np.sin(beta)
        a_y = v_dot * np.sin(beta) - v * (r + beta_dot) * np.cos(beta)

        return [x_dot, y_dot, psi_dot, beta_dot, r_dot, a_x, a_y, delta]

    def system_dynamics(self, vars_, t, v, delta):  # vars_dot = f(vars,t)
        x_dot, y_dot, psi_dot, beta_dot, r_dot, _, _, _ = self.system_results(
            vars_, v, delta
        )

        dvarsdt = [x_dot, y_dot, psi_dot, beta_dot, r_dot]
        return dvarsdt

    def system_outputs(self, vars_, v, delta):
        _, _, _, _, _, a_x, a_y, delta_pt1 = self.system_results(vars_, v, delta)
        return [a_x, a_y, delta_pt1]

    def simulate(self, x0, v, delta, ti):
        sol = odeint(self.system_dynamics, x0, ti, args=(v, delta))
        return sol
