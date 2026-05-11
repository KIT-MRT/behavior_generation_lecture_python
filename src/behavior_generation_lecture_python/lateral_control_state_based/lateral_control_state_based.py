"""Lateral vehicle control using state-based feedback with kinematic model."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import odeint

import behavior_generation_lecture_python.lateral_control_state_based.state_based_controller as con
import behavior_generation_lecture_python.utils.projection as pro
import behavior_generation_lecture_python.vehicle_models.kinematic_one_track_model as kotm
from behavior_generation_lecture_python.utils.reference_curve import ReferenceCurve


@dataclass
class KinematicVehicleState:
    """State of a vehicle using the kinematic one-track model.

    Attributes:
        x: X-position in global coordinates [m]
        y: Y-position in global coordinates [m]
        heading: Vehicle heading angle (yaw) [rad]
    """

    x: float
    y: float
    heading: float

    def to_list(self) -> list[float]:
        """Convert to list for use with numerical integrators."""
        return [self.x, self.y, self.heading]


@dataclass
class ControllerOutput:
    """Output of the state-based lateral controller at a single time step.

    Attributes:
        x: Vehicle x-position [m]
        y: Vehicle y-position [m]
        heading: Vehicle heading angle [rad]
        lateral_error: Distance from vehicle to reference curve [m]
        steering_angle: Commanded steering angle [rad]
    """

    x: float
    y: float
    heading: float
    lateral_error: float
    steering_angle: float


class LateralControlStateBased:
    """Lateral vehicle controller using state-based feedback with kinematic model.

    This controller uses a simple state feedback design based on a kinematic
    one-track (bicycle) model. It computes steering commands based on:
    - Lateral error (distance to reference curve)
    - Heading error (difference from reference heading)
    - Curvature feedforward for steady-state cornering
    """

    def __init__(
        self,
        initial_state: KinematicVehicleState,
        curve: ReferenceCurve,
    ):
        """Initialize the state-based lateral controller.

        Args:
            initial_state: Initial vehicle state (position and heading)
            curve: Reference curve to follow
        """
        self.initial_state = initial_state
        self.reference_curve = curve
        self.velocity = 1.0

    def simulate(
        self, time_vector: np.ndarray[Any, Any], velocity: float = 1.0
    ) -> list[ControllerOutput]:
        """Simulate the closed-loop vehicle trajectory.

        Args:
            time_vector: Array of time points for simulation [s]
            velocity: Constant longitudinal velocity [m/s]

        Returns:
            List of ControllerOutput for each time step
        """
        self.velocity = velocity
        state_trajectory = odeint(
            self._compute_state_derivatives, self.initial_state.to_list(), time_vector
        )
        return [self._compute_output(state) for state in state_trajectory]

    def _compute_state_derivatives(
        self, state: np.ndarray[Any, Any], time: float
    ) -> np.ndarray[Any, Any]:
        """Compute state derivatives for the closed-loop system.

        Args:
            state: Current state [x, y, psi]
            time: Current simulation time [s]

        Returns:
            State derivatives [x_dot, y_dot, psi_dot]
        """
        x, y, psi = state
        projection = pro.project2curve(
            self.reference_curve.arc_length,
            self.reference_curve.x,
            self.reference_curve.y,
            self.reference_curve.heading,
            self.reference_curve.curvature,
            x,
            y,
        )
        steering_angle = con.feedback_law(
            projection.lateral_error, psi, projection.heading, projection.curvature
        )
        # Vehicle model is not fully typed yet
        state_derivatives: np.ndarray[Any, Any] = kotm.KinematicOneTrackModel().system_dynamics(  # type: ignore[no-untyped-call]
            state, time, self.velocity, steering_angle
        )
        return state_derivatives

    def _compute_output(self, state: np.ndarray[Any, Any]) -> ControllerOutput:
        """Compute output variables for the current state.

        Args:
            state: Current state [x, y, psi]

        Returns:
            ControllerOutput with position, heading, lateral error, and steering angle
        """
        x, y, psi = state
        projection = pro.project2curve(
            self.reference_curve.arc_length,
            self.reference_curve.x,
            self.reference_curve.y,
            self.reference_curve.heading,
            self.reference_curve.curvature,
            x,
            y,
        )
        steering_angle = con.feedback_law(
            projection.lateral_error, psi, projection.heading, projection.curvature
        )

        return ControllerOutput(
            x=x,
            y=y,
            heading=psi,
            lateral_error=projection.lateral_error,
            steering_angle=steering_angle,
        )
