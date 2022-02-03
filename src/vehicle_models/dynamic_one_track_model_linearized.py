import numpy as np
from scipy.integrate import odeint

from vehicle_models.vehicle_parameters import DEFAULT_VEHICLE_PARAMS


class DynamicOneTrackModelLinearized:
    def __init__(self):
        self.params = DEFAULT_VEHICLE_PARAMS

    def system_results(self, vars_, v, delta):
        beta, r = vars_

        c_v = self.params.A_v * self.params.B_v * self.params.C_v
        c_h = self.params.A_h * self.params.B_h * self.params.C_h

        x = np.array([beta, r])
        u = delta

        a11 = -(c_h + c_v) / (self.params.m * v)
        a12 = -1 + (c_h * self.params.l_h - c_v * self.params.l_v) / (
            self.params.m * np.power(v, 2)
        )
        a21 = (c_h * self.params.l_h - c_v * self.params.l_v) / self.params.J
        a22 = -(
            c_v * np.power(self.params.l_v, 2) + c_h * np.power(self.params.l_h, 2)
        ) / (self.params.J * v)

        A = np.array([[a11, a12], [a21, a22]])
        b = np.array([c_v / (self.params.m * v), c_v * self.params.l_v / self.params.J])

        vars_dot = np.matmul(A, x) + b * u
        beta_dot = vars_dot[0]
        r_dot = vars_dot[1]

        return [beta_dot, r_dot, delta]

    def system_dynamics(self, vars_, t, v, delta):  # vars_dot = f(vars,t)
        beta_dot, r_dot, _ = self.system_results(vars_, v, delta)

        dvarsdt = [beta_dot, r_dot]
        return dvarsdt

    def system_outputs(self, vars_, v, delta):
        _, _, delta = self.system_results(vars_, v, delta)
        return [delta]

    def simulate(self, x0, v, delta, ti):
        sol = odeint(self.system_dynamics, x0, ti, args=(v, delta))
        return sol
