"""Lateral vehicle control using LQR (Linear Quadratic Regulator) with Riccati equation."""

import math
from dataclasses import dataclass

import numpy as np
import scipy.linalg  # type: ignore[import-untyped]
from scipy import signal  # type: ignore[import-untyped]
from scipy.integrate import odeint  # type: ignore[import-untyped]

import behavior_generation_lecture_python.lateral_control_riccati.riccati_controller as con
import behavior_generation_lecture_python.utils.projection as pro
import behavior_generation_lecture_python.vehicle_models.dynamic_one_track_model as dotm
from behavior_generation_lecture_python.utils.reference_curve import ReferenceCurve
from behavior_generation_lecture_python.vehicle_models.vehicle_parameters import (
    VehicleParameters,
)


@dataclass
class ControlParameters:
    """Parameters for the LQR lateral controller.

    Attributes:
        lookahead_distance: Look-ahead distance for the controller [m]
        lqr_gain: LQR feedback gain vector [k_lateral, k_heading, k_beta, k_yaw_rate]
        disturbance_compensation_gain: Gain for curvature feedforward compensation
    """

    lookahead_distance: float
    lqr_gain: np.ndarray
    disturbance_compensation_gain: float


@dataclass
class LQRSolution:
    """Solution of the continuous-time LQR problem.

    Attributes:
        feedback_gain: State feedback gain vector K
        riccati_solution: Solution matrix X of the algebraic Riccati equation
        closed_loop_eigenvalues: Eigenvalues of the closed-loop system (A - BK)
    """

    feedback_gain: np.ndarray
    riccati_solution: np.ndarray
    closed_loop_eigenvalues: np.ndarray


@dataclass
class ActuatorDynamicsOutput:
    """Output of the PT2 actuator dynamics computation.

    Attributes:
        steering_angle_derivative: Rate of change of steering angle [rad/s]
        steering_rate_derivative: Rate of change of steering rate [rad/s^2]
        actual_steering_angle: Current actual steering angle after actuator dynamics [rad]
    """

    steering_angle_derivative: float
    steering_rate_derivative: float
    actual_steering_angle: float


@dataclass
class DynamicVehicleState:
    """State of a vehicle using the dynamic one-track model.

    Attributes:
        x: X-position in global coordinates [m]
        y: Y-position in global coordinates [m]
        heading: Vehicle heading angle (yaw) [rad]
        sideslip_angle: Sideslip angle (beta) at center of gravity [rad]
        yaw_rate: Angular velocity around vertical axis [rad/s]
    """

    x: float
    y: float
    heading: float
    sideslip_angle: float
    yaw_rate: float

    def to_list(self) -> list[float]:
        """Convert to list for use with numerical integrators."""
        return [self.x, self.y, self.heading, self.sideslip_angle, self.yaw_rate]


@dataclass
class SimulationState:
    """Full simulation state including vehicle dynamics and actuator states.

    Attributes:
        x: X-position in global coordinates [m]
        y: Y-position in global coordinates [m]
        heading: Vehicle heading angle (yaw) [rad]
        sideslip_angle: Sideslip angle (beta) at center of gravity [rad]
        yaw_rate: Angular velocity around vertical axis [rad/s]
        steering_angle: Current steering angle [rad]
        steering_rate: Current rate of change of steering angle [rad/s]
    """

    x: float
    y: float
    heading: float
    sideslip_angle: float
    yaw_rate: float
    steering_angle: float
    steering_rate: float

    @classmethod
    def from_vehicle_state(
        cls,
        vehicle_state: DynamicVehicleState,
        steering_angle: float = 0.0,
        steering_rate: float = 0.0,
    ) -> "SimulationState":
        """Create simulation state from vehicle state with initial actuator values."""
        return cls(
            x=vehicle_state.x,
            y=vehicle_state.y,
            heading=vehicle_state.heading,
            sideslip_angle=vehicle_state.sideslip_angle,
            yaw_rate=vehicle_state.yaw_rate,
            steering_angle=steering_angle,
            steering_rate=steering_rate,
        )

    def to_list(self) -> list[float]:
        """Convert to list for use with numerical integrators."""
        return [
            self.x,
            self.y,
            self.heading,
            self.sideslip_angle,
            self.yaw_rate,
            self.steering_angle,
            self.steering_rate,
        ]


@dataclass
class StateDerivatives:
    """Time derivatives of the simulation state.

    Attributes:
        x_dot: Velocity in x-direction [m/s]
        y_dot: Velocity in y-direction [m/s]
        heading_dot: Yaw rate [rad/s]
        sideslip_angle_dot: Rate of change of sideslip angle [rad/s]
        yaw_rate_dot: Angular acceleration [rad/s^2]
        steering_angle_dot: Rate of change of steering angle [rad/s]
        steering_rate_dot: Steering acceleration [rad/s^2]
    """

    x_dot: float
    y_dot: float
    heading_dot: float
    sideslip_angle_dot: float
    yaw_rate_dot: float
    steering_angle_dot: float
    steering_rate_dot: float

    def to_tuple(self) -> tuple[float, float, float, float, float, float, float]:
        """Convert to tuple for use with numerical integrators like odeint."""
        return (
            self.x_dot,
            self.y_dot,
            self.heading_dot,
            self.sideslip_angle_dot,
            self.yaw_rate_dot,
            self.steering_angle_dot,
            self.steering_rate_dot,
        )


def get_control_params(
    vehicle_params: VehicleParameters, velocity: float, control_weight: float
) -> ControlParameters:
    """Compute LQR control parameters for the given vehicle and velocity.

    This function linearizes the vehicle dynamics at the given velocity and
    solves the LQR problem to obtain optimal feedback gains.

    Args:
        vehicle_params: Physical parameters of the vehicle
        velocity: Longitudinal velocity for linearization [m/s]
        control_weight: Weight on control effort in LQR cost (higher = less aggressive)

    Returns:
        ControlParameters containing LQR gains and disturbance compensation.
    """
    # Compute cornering stiffness from tire parameters (Pacejka magic formula coefficients)
    cornering_stiffness_front = (
        vehicle_params.A_v * vehicle_params.B_v * vehicle_params.C_v
    )
    cornering_stiffness_rear = (
        vehicle_params.A_h * vehicle_params.B_h * vehicle_params.C_h
    )

    # Linearized lateral dynamics matrix elements
    a11 = -(cornering_stiffness_rear + cornering_stiffness_front) / (
        vehicle_params.m * velocity
    )
    a12 = -1 + (
        cornering_stiffness_rear * vehicle_params.l_h
        - cornering_stiffness_front * vehicle_params.l_v
    ) / (vehicle_params.m * np.power(velocity, 2))
    a21 = (
        cornering_stiffness_rear * vehicle_params.l_h
        - cornering_stiffness_front * vehicle_params.l_v
    ) / vehicle_params.J
    a22 = -(
        cornering_stiffness_front * np.power(vehicle_params.l_v, 2)
        + cornering_stiffness_rear * np.power(vehicle_params.l_h, 2)
    ) / (vehicle_params.J * velocity)

    A_lateral_dynamics = np.array([[a11, a12], [a21, a22]])
    B_lateral_dynamics = np.array(
        [
            cornering_stiffness_front / (vehicle_params.m * velocity),
            cornering_stiffness_front * vehicle_params.l_v / vehicle_params.J,
        ]
    )

    # Augmented system matrix for error dynamics
    # State: [lateral_error, heading_error, beta, yaw_rate]
    A_augmented = np.array(
        [
            [0, velocity, velocity, vehicle_params.l_s],
            [0, 0, 0, 1],
            [0, 0, A_lateral_dynamics[0, 0], A_lateral_dynamics[0, 1]],
            [0, 0, A_lateral_dynamics[1, 0], A_lateral_dynamics[1, 1]],
        ]
    )
    B_augmented = (
        np.array([0, 0, B_lateral_dynamics[0], B_lateral_dynamics[1]])[np.newaxis]
    ).transpose()

    # LQR state weighting matrix (identity = equal weight on all states)
    Q_state_weight = np.zeros((4, 4))
    np.fill_diagonal(Q_state_weight, 1)

    lqr_solution = lqr(A=A_augmented, B=B_augmented, Q=Q_state_weight, R=control_weight)

    # Compute disturbance compensation gain (understeer gradient compensation)
    wheelbase = vehicle_params.l_h + vehicle_params.l_v
    understeer_gradient = (
        vehicle_params.m
        / wheelbase
        * (
            vehicle_params.l_h / cornering_stiffness_front
            - vehicle_params.l_v / cornering_stiffness_rear
        )
    )
    disturbance_compensation_gain = wheelbase + understeer_gradient * np.power(
        velocity, 2
    )

    return ControlParameters(
        lookahead_distance=vehicle_params.l_s,
        lqr_gain=lqr_solution.feedback_gain,
        disturbance_compensation_gain=disturbance_compensation_gain,
    )


def lqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: float) -> LQRSolution:
    """Solve the continuous-time Linear Quadratic Regulator (LQR) problem.

    Finds the optimal state-feedback gain K that minimizes the cost function:
    J = integral(x'Qx + u'Ru) dt

    Args:
        A: System dynamics matrix (n x n)
        B: Input matrix (n x 1)
        Q: State weighting matrix (n x n), must be positive semi-definite
        R: Control weighting scalar, must be positive

    Returns:
        LQRSolution containing feedback gain, Riccati solution, and closed-loop eigenvalues.
    """
    riccati_solution = scipy.linalg.solve_continuous_are(A, B, Q, R)

    feedback_gain_matrix = (1 / R) * np.dot(B.T, riccati_solution)
    feedback_gain = np.array(
        [
            feedback_gain_matrix[0, 0],
            feedback_gain_matrix[0, 1],
            feedback_gain_matrix[0, 2],
            feedback_gain_matrix[0, 3],
        ]
    )

    closed_loop_eigenvalues, _ = scipy.linalg.eig(A - B * feedback_gain)

    return LQRSolution(
        feedback_gain=feedback_gain,
        riccati_solution=riccati_solution,
        closed_loop_eigenvalues=closed_loop_eigenvalues,
    )


class LateralControlRiccati:
    """Lateral vehicle controller using LQR with dynamic one-track model.

    This controller uses a Linear Quadratic Regulator (LQR) design based on a
    linearized dynamic one-track (bicycle) model. It includes:
    - State feedback for lateral error, heading error, sideslip, and yaw rate
    - Curvature feedforward compensation for steady-state cornering
    - PT2 actuator dynamics for realistic steering response
    - Measurement noise simulation
    """

    def __init__(
        self,
        initial_state: DynamicVehicleState,
        curve: ReferenceCurve,
        vehicle_params: VehicleParameters,
        initial_velocity: float,
        control_weight: float,
    ):
        """Initialize the LQR lateral controller.

        Args:
            initial_state: Initial vehicle state (position, heading, sideslip, yaw rate)
            curve: Reference curve to follow
            vehicle_params: Physical parameters of the vehicle
            initial_velocity: Initial longitudinal velocity [m/s]
            control_weight: LQR control weight (higher = less aggressive steering)
        """
        self.initial_simulation_state = SimulationState.from_vehicle_state(
            initial_state, steering_angle=0.0, steering_rate=0.0
        )
        self.reference_curve = curve
        self.velocity = initial_velocity
        self.vehicle_params = vehicle_params
        self.control_params = get_control_params(
            vehicle_params=vehicle_params,
            velocity=initial_velocity,
            control_weight=control_weight,
        )
        # PT2 actuator dynamics (second-order low-pass filter for steering)
        actuator_time_constant = 0.05
        numerator = [1]
        denominator = [
            2 * np.power(actuator_time_constant, 2),
            2 * actuator_time_constant,
            1,
        ]
        self.actuator_state_space = signal.TransferFunction(
            numerator, denominator
        ).to_ss()

    def simulate(
        self, time_vector: np.ndarray, velocity: float = 1, time_step: float = 0.1
    ) -> np.ndarray:
        """Simulate the closed-loop vehicle trajectory.

        Args:
            time_vector: Array of time points for simulation [s]
            velocity: Constant longitudinal velocity [m/s]
            time_step: Time step for noise generation [s]

        Returns:
            State trajectory array with shape (len(time_vector), 7).
            Columns: [x, y, psi, beta, yaw_rate, steering_angle, steering_rate]
        """
        self.velocity = velocity
        state_trajectory = odeint(
            self._compute_state_derivatives,
            self.initial_simulation_state.to_list(),
            time_vector,
            args=(time_step,),
        )
        return state_trajectory

    @staticmethod
    def _add_position_noise(value: float, seed: int) -> float:
        """Add Gaussian noise to simulate position measurement uncertainty.

        Args:
            value: True position value
            seed: Random seed for reproducibility

        Returns:
            Noisy position value
        """
        position_noise_std = 0.01  # meters
        np.random.seed(seed)
        noise = np.random.normal(0, position_noise_std)
        return value + noise

    @staticmethod
    def _add_orientation_noise(value: float, seed: int) -> float:
        """Add Gaussian noise to simulate orientation measurement uncertainty.

        Args:
            value: True orientation value [rad]
            seed: Random seed for reproducibility

        Returns:
            Noisy orientation value [rad]
        """
        orientation_noise_std = 1.0 / 180 * math.pi  # 1 degree in radians
        np.random.seed(seed)
        noise = np.random.normal(0, orientation_noise_std)
        return value + noise

    def _compute_actuator_dynamics(
        self,
        steering_angle: float,
        steering_rate: float,
        steering_command: float,
    ) -> ActuatorDynamicsOutput:
        """Compute PT2 actuator dynamics for the steering system.

        Args:
            steering_angle: Current steering angle [rad]
            steering_rate: Current steering rate [rad/s]
            steering_command: Commanded steering angle [rad]

        Returns:
            ActuatorDynamicsOutput with derivatives and actual steering angle
        """
        state = np.array([[steering_angle], [steering_rate]])
        state_derivative = np.dot(self.actuator_state_space.A, state) + np.dot(
            self.actuator_state_space.B, steering_command
        )
        actual_steering = np.dot(self.actuator_state_space.C, state) + np.dot(
            self.actuator_state_space.D, steering_command
        )
        return ActuatorDynamicsOutput(
            steering_angle_derivative=state_derivative[0, 0],
            steering_rate_derivative=state_derivative[1, 0],
            actual_steering_angle=actual_steering[0, 0],
        )

    def _compute_state_derivatives(
        self, state: np.ndarray, time: float, time_step: float
    ) -> tuple[float, float, float, float, float, float, float]:
        """Compute state derivatives for the closed-loop system.

        Args:
            state: Current state [x, y, psi, beta, yaw_rate, steering_angle, steering_rate]
            time: Current simulation time [s]
            time_step: Time step for noise generation [s]

        Returns:
            State derivatives as tuple (required by odeint)
        """
        current_state = SimulationState(
            x=state[0],
            y=state[1],
            heading=state[2],
            sideslip_angle=state[3],
            yaw_rate=state[4],
            steering_angle=state[5],
            steering_rate=state[6],
        )

        # Project vehicle position onto reference curve with look-ahead
        projection = pro.project2curve_with_lookahead(
            self.reference_curve.arc_length,
            self.reference_curve.x,
            self.reference_curve.y,
            self.reference_curve.heading,
            self.reference_curve.curvature,
            self.control_params.lookahead_distance,
            current_state.x,
            current_state.y,
            current_state.heading,
        )

        # Add measurement noise (synchronized by time step)
        noise_seed = math.floor(time / time_step)
        lateral_error = self._add_position_noise(projection.lateral_error, noise_seed)
        heading_error = self._add_orientation_noise(projection.heading, noise_seed)

        # Compute steering command from feedback law
        steering_command = con.feedback_law(
            self.control_params.lqr_gain,
            self.control_params.disturbance_compensation_gain,
            lateral_error,
            heading_error,
            projection.curvature,
            current_state.sideslip_angle,
            current_state.yaw_rate,
        )

        # Apply actuator dynamics
        actuator_output = self._compute_actuator_dynamics(
            current_state.steering_angle,
            current_state.steering_rate,
            steering_command,
        )

        # Compute vehicle dynamics
        vehicle_state = state[:5]
        vehicle_derivatives = dotm.DynamicOneTrackModel(
            self.vehicle_params
        ).system_dynamics(
            vehicle_state, time, self.velocity, actuator_output.actual_steering_angle
        )

        derivatives = StateDerivatives(
            x_dot=vehicle_derivatives[0],
            y_dot=vehicle_derivatives[1],
            heading_dot=vehicle_derivatives[2],
            sideslip_angle_dot=vehicle_derivatives[3],
            yaw_rate_dot=vehicle_derivatives[4],
            steering_angle_dot=actuator_output.steering_angle_derivative,
            steering_rate_dot=actuator_output.steering_rate_derivative,
        )
        return derivatives.to_tuple()
