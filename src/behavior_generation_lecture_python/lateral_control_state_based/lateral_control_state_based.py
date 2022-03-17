import numpy as np
from scipy.integrate import odeint

import behavior_generation_lecture_python.lateral_control_state_based.state_based_controller as con
import behavior_generation_lecture_python.utils.projection as pro
import behavior_generation_lecture_python.vehicle_models.kinematic_one_track_model as kotm


class LateralControlStateBased:
    def __init__(self, initial_condition, curve):
        self.vars_0 = initial_condition
        self.curve = curve
        self.v = 1

    def simulate(self, t_vector, v=1):
        self.v = v
        state_trajectory = odeint(self._f_system_dynamics, self.vars_0, t_vector)
        output_trajectory = np.array(
            [self._g_system_output(x) for x in state_trajectory]
        )
        return output_trajectory

    def _f_system_dynamics(self, vars_, t):
        x, y, psi = vars_
        _, _, _, d, theta_r, kappa_r = pro.project2curve(
            self.curve["s"],
            self.curve["x"],
            self.curve["y"],
            self.curve["theta"],
            self.curve["kappa"],
            x,
            y,
        )
        delta = con.feedback_law(d, psi, theta_r, kappa_r)
        v = self.v  # const velocity
        vars_dot = kotm.KinematicOneTrackModel().system_dynamics(vars_, t, v, delta)
        return vars_dot

    def _g_system_output(self, vars_):
        x, y, psi = vars_
        _, _, _, d, theta_r, kappa_r = pro.project2curve(
            self.curve["s"],
            self.curve["x"],
            self.curve["y"],
            self.curve["theta"],
            self.curve["kappa"],
            x,
            y,
        )
        delta = con.feedback_law(d, psi, theta_r, kappa_r)

        return [x, y, psi, d, delta]
