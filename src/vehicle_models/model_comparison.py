import numpy as np
from scipy.integrate import odeint
import vehicle_models.dynamic_one_track_model as nlotm
import vehicle_models.dynamic_one_track_model_linearized as lotm


class CompareModels:
    def __init__(self, initial_condition, delta_fun):
        self.vars_0 = initial_condition
        self.delta_fun = delta_fun
        self.v = 1

    def simulate(self, t_vector, v=1):
        self.v = v
        state_trajectory = odeint(
            self._f_system_dynamics_non_linear, self.vars_0, t_vector
        )
        state_linear = odeint(self._f_system_dynamics_linear, self.vars_0[3:], t_vector)
        output_trajectory = np.array([x for x in state_trajectory])
        output_linear = np.array([x for x in state_linear])
        return [output_trajectory, output_linear]

    def _f_system_dynamics_non_linear(self, vars_, t):
        delta = self.delta_fun(t)
        v = self.v  # const velocity
        vars_dot = nlotm.DynamicOneTrackModel().system_dynamics(vars_, t, v, delta)
        return vars_dot

    def _f_system_dynamics_linear(self, vars_, t):
        delta = self.delta_fun(t)
        v = self.v  # const velocity
        vars_dot = lotm.DynamicOneTrackModelLinearized().system_dynamics(
            vars_, t, v, delta
        )
        return vars_dot
