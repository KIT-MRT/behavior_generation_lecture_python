import math
import numpy as np
import scipy.linalg
from scipy import signal
from scipy.integrate import odeint
from dataclasses import dataclass

import lateral_control_riccati.riccati_controller as con
import utils.projection as pro
import vehicle_models.dynamic_one_track_model as dotm
from vehicle_models.vehicle_parameters import VehicleParameters


@dataclass
class ControlParameters:
    l_s: float
    k_lqr: np.ndarray
    k_dist_comp: float


def get_control_params(vehicle_params: VehicleParameters, velocity: float, r: float):
    v_0 = velocity

    c_v = vehicle_params.A_v * vehicle_params.B_v * vehicle_params.C_v
    c_h = vehicle_params.A_h * vehicle_params.B_h * vehicle_params.C_h

    a11 = -(c_h + c_v) / (vehicle_params.m * v_0)
    a12 = -1 + (c_h * vehicle_params.l_h - c_v * vehicle_params.l_v) / (
        vehicle_params.m * np.power(v_0, 2)
    )
    a21 = (c_h * vehicle_params.l_h - c_v * vehicle_params.l_v) / vehicle_params.J
    a22 = -(
        c_v * np.power(vehicle_params.l_v, 2) + c_h * np.power(vehicle_params.l_h, 2)
    ) / (vehicle_params.J * v_0)

    A_lOTM = np.array([[a11, a12], [a21, a22]])
    b_lOTM = np.array(
        [
            c_v / (vehicle_params.m * v_0),
            c_v * vehicle_params.l_v / vehicle_params.J,
        ]
    )

    A = np.array(
        [
            [0, v_0, v_0, vehicle_params.l_s],
            [0, 0, 0, 1],
            [0, 0, A_lOTM[0, 0], A_lOTM[0, 1]],
            [0, 0, A_lOTM[1, 0], A_lOTM[1, 1]],
        ]
    )
    b = (np.array([0, 0, b_lOTM[0], b_lOTM[1]])[np.newaxis]).transpose()

    Q = np.zeros((4, 4))
    np.fill_diagonal(Q, 1)

    k_lqr, _, _ = lqr(A=A, b=b, Q=Q, r=r)

    l = vehicle_params.l_h + vehicle_params.l_v
    EG = vehicle_params.m / l * (vehicle_params.l_h / c_v - vehicle_params.l_v / c_h)
    k_dist_comp = l + EG * np.power(v_0, 2)

    return ControlParameters(
        l_s=vehicle_params.l_s, k_lqr=k_lqr, k_dist_comp=k_dist_comp
    )


def lqr(A, b, Q, r):
    X = scipy.linalg.solve_continuous_are(A, b, Q, r)

    K = (1 / r) * np.dot(b.T, X)
    K = np.array([K[0, 0], K[0, 1], K[0, 2], K[0, 3]])

    eig_vals, eig_vecs = scipy.linalg.eig(A - b * K)

    return K, X, eig_vals


class LateralControlRiccati:
    def __init__(
        self,
        initial_condition: np.array,
        curve: dict,
        vehicle_params: VehicleParameters,
        initial_velocity: float,
        r: float,
    ):
        self.vars_0 = initial_condition
        self.vars_0.append(0.0)
        self.vars_0.append(0.0)
        self.curve = curve
        self.v = initial_velocity
        self.vehicle_params = vehicle_params
        self.params = get_control_params(
            vehicle_params=vehicle_params, velocity=initial_velocity, r=r
        )
        num = [1]
        den = [2 * np.power(0.05, 2), 2 * 0.05, 1]
        self.tf_ss = signal.TransferFunction(num, den).to_ss()

    def simulate(self, t_vector, v=1, t_step=0.1):
        self.v = v
        state_trajectory = odeint(
            self._f_system_dynamics, self.vars_0, t_vector, args=(t_step,)
        )
        return state_trajectory

    @staticmethod
    def __position_noise(val, seed):
        position_noise = 0.01
        mu = 0
        sigma = position_noise

        np.random.seed(seed)
        noise = np.random.normal(mu, sigma)
        result = val + noise

        return result

    @staticmethod
    def __orientation_noise(val, seed):
        orientation_noise = 1.0 / 180 * math.pi
        mu = 0
        sigma = orientation_noise

        np.random.seed(seed)
        noise = np.random.normal(mu, sigma)
        result = val + noise

        return result

    def __pt2_motor_dynamic(self, vars_, t, delta_in):
        state_1, state_2 = vars_
        state = np.matrix([state_1, state_2]).T
        dvarsdt = np.dot(self.tf_ss.A, state) + np.dot(self.tf_ss.B, delta_in)
        delta = np.dot(self.tf_ss.C, state) + np.dot(self.tf_ss.D, delta_in)
        return dvarsdt[0, 0], dvarsdt[1, 0], delta[0, 0]

    @staticmethod
    def __delta_pt1(delta):
        delta = delta * 18
        return delta

    def _f_system_dynamics(self, vars_, t, t_step):
        x, y, psi, beta, r, delta, delta_dot = vars_
        _, _, _, e_l, e_psi, kappa_r = pro.project2curve_with_lookahead(
            self.curve["s"],
            self.curve["x"],
            self.curve["y"],
            self.curve["theta"],
            self.curve["kappa"],
            self.params.l_s,
            x,
            y,
            psi,
        )
        seed = math.floor(t / t_step)
        e_l = self.__position_noise(e_l, seed)
        e_psi = self.__orientation_noise(e_psi, seed)
        delta_in = con.feedback_law(
            self.params.k_lqr,
            self.params.k_dist_comp,
            e_l,
            e_psi,
            kappa_r,
            beta,
            r,
        )
        state_1_dot, state_2_dot, delta = self.__pt2_motor_dynamic(
            [delta, delta_dot], t, delta_in
        )
        v = self.v  # const velocity
        vars_dot = dotm.DynamicOneTrackModel(self.vehicle_params).system_dynamics(
            vars_[:5], t, v, delta
        )
        return (
            vars_dot[0],
            vars_dot[1],
            vars_dot[2],
            vars_dot[3],
            vars_dot[4],
            state_1_dot,
            state_2_dot,
        )
