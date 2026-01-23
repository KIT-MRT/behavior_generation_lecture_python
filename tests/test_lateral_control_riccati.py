import numpy as np
import pytest

import behavior_generation_lecture_python.lateral_control_riccati.lateral_control_riccati as cl
import behavior_generation_lecture_python.lateral_control_riccati.riccati_controller as con
import behavior_generation_lecture_python.utils.generate_reference_curve as ref
from behavior_generation_lecture_python.lateral_control_riccati.lateral_control_riccati import (
    DynamicVehicleState,
)
from behavior_generation_lecture_python.utils.projection import project2curve
from behavior_generation_lecture_python.vehicle_models.vehicle_parameters import (
    DEFAULT_VEHICLE_PARAMS,
)


@pytest.mark.parametrize("control_weight,error_factor", [(10000, 1), (10, 0.5)])
def test_lateral_control_riccati(control_weight, error_factor):
    radius = 500
    initial_state = DynamicVehicleState(
        x=0.0,
        y=-radius,
        heading=0.0,
        sideslip_angle=0.0,
        yaw_rate=0.0,
    )
    initial_velocity = 33.0

    curve = ref.generate_reference_curve(
        np.array([0, radius, 0, -radius, 0]),
        np.array([-radius, 0, radius, 0, radius]),
        10.0,
    )
    time_vector = np.arange(0, 40, 0.1)

    model = cl.LateralControlRiccati(
        initial_state=initial_state,
        curve=curve,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        initial_velocity=initial_velocity,
        control_weight=control_weight,
    )
    trajectory = model.simulate(time_vector, velocity=initial_velocity, time_step=0.1)

    errors = []
    for state in trajectory:
        projection = project2curve(
            s_c=curve.arc_length,
            x_c=curve.x,
            y_c=curve.y,
            theta_c=curve.heading,
            kappa_c=curve.curvature,
            x=state[0],
            y=state[1],
        )
        errors.append(abs(projection.lateral_error))

    assert np.sum(errors) > len(trajectory) * 0.1 * error_factor
    assert np.sum(errors) < len(trajectory) * 0.2 * error_factor


# Unit tests for riccati_controller.feedback_law()
class TestFeedbackLaw:
    def test_feedback_law_zero_error(self):
        """Zero steering when on reference with no curvature and no dynamics"""
        k_lqr = np.array([1.0, 1.0, 1.0, 1.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=1.0,
            lateral_error=0,
            heading_error=0,
            reference_curvature=0,
            beta=0,
            yaw_rate=0,
        )
        assert steering == pytest.approx(0)

    def test_feedback_law_lateral_error_positive(self):
        """Negative steering to correct positive lateral error (left of reference)"""
        k_lqr = np.array([1.0, 0.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=1.0,
            heading_error=0,
            reference_curvature=0,
            beta=0,
            yaw_rate=0,
        )
        assert steering < 0, "Should steer right (negative) to correct left error"

    def test_feedback_law_lateral_error_negative(self):
        """Positive steering to correct negative lateral error (right of reference)"""
        k_lqr = np.array([1.0, 0.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=-1.0,
            heading_error=0,
            reference_curvature=0,
            beta=0,
            yaw_rate=0,
        )
        assert steering > 0, "Should steer left (positive) to correct right error"

    def test_feedback_law_heading_error_positive(self):
        """Steering correction for positive heading error"""
        k_lqr = np.array([0.0, 1.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=0,
            heading_error=0.1,
            reference_curvature=0,
            beta=0,
            yaw_rate=0,
        )
        assert steering < 0, "Should steer right to correct heading pointing left"

    def test_feedback_law_heading_error_negative(self):
        """Steering correction for negative heading error"""
        k_lqr = np.array([0.0, 1.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=0,
            heading_error=-0.1,
            reference_curvature=0,
            beta=0,
            yaw_rate=0,
        )
        assert steering > 0, "Should steer left to correct heading pointing right"

    def test_feedback_law_curvature_feedforward_positive(self):
        """Positive curvature should add positive steering component"""
        k_lqr = np.array([0.0, 0.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=1.0,
            lateral_error=0,
            heading_error=0,
            reference_curvature=0.1,
            beta=0,
            yaw_rate=0,
        )
        assert steering > 0, "Positive curvature should cause positive steering"

    def test_feedback_law_curvature_feedforward_negative(self):
        """Negative curvature should add negative steering component"""
        k_lqr = np.array([0.0, 0.0, 0.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=1.0,
            lateral_error=0,
            heading_error=0,
            reference_curvature=-0.1,
            beta=0,
            yaw_rate=0,
        )
        assert steering < 0, "Negative curvature should cause negative steering"

    def test_feedback_law_beta_correction(self):
        """Sideslip angle (beta) should be corrected"""
        k_lqr = np.array([0.0, 0.0, 1.0, 0.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=0,
            heading_error=0,
            reference_curvature=0,
            beta=0.1,
            yaw_rate=0,
        )
        assert steering < 0, "Positive beta should cause negative steering correction"

    def test_feedback_law_yaw_rate_correction(self):
        """Yaw rate should be corrected"""
        k_lqr = np.array([0.0, 0.0, 0.0, 1.0])
        steering = con.feedback_law(
            k_lqr=k_lqr,
            k_dist_comp=0.0,
            lateral_error=0,
            heading_error=0,
            reference_curvature=0,
            beta=0,
            yaw_rate=0.1,
        )
        assert (
            steering < 0
        ), "Positive yaw rate should cause negative steering correction"


# Unit tests for lqr() function
# Note: The lqr() function is specifically designed for 4-state systems
class TestLQR:
    @pytest.fixture
    def sample_4d_system(self):
        """A sample 4D system similar to vehicle lateral control"""
        A = np.array([[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, -1, 0.5], [0, 0, 0.5, -1]])
        B = np.array([[0], [0], [1], [0.5]])
        Q = np.eye(4)
        return A, B, Q

    def test_lqr_4d_system_stability(self, sample_4d_system):
        """Test LQR on 4D system - closed loop should be stable"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        # Verify closed-loop eigenvalues are stable (negative real parts)
        assert all(
            np.real(solution.closed_loop_eigenvalues) < 0
        ), "Closed-loop system should have stable eigenvalues"

    def test_lqr_gain_shape(self, sample_4d_system):
        """Verify feedback_gain has correct shape for 4D system"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        assert solution.feedback_gain.shape == (
            4,
        ), "K should have 4 elements for 4D system"

    def test_lqr_riccati_solution_shape(self, sample_4d_system):
        """Verify riccati_solution has correct shape"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        assert solution.riccati_solution.shape == (
            4,
            4,
        ), "X should be 4x4 for 4D system"

    def test_lqr_riccati_solution_symmetric(self, sample_4d_system):
        """Verify riccati_solution is symmetric"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        assert np.allclose(
            solution.riccati_solution, solution.riccati_solution.T
        ), "Riccati solution should be symmetric"

    def test_lqr_riccati_solution_positive_semidefinite(self, sample_4d_system):
        """Verify riccati_solution is positive semi-definite"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        eigenvalues_X = np.linalg.eigvals(solution.riccati_solution)
        assert all(
            eigenvalues_X >= -1e-10
        ), "Riccati solution should be positive semi-definite"

    def test_lqr_higher_control_cost_smaller_gains(self, sample_4d_system):
        """Higher control cost R should result in smaller gains"""
        A, B, Q = sample_4d_system
        solution_low_R = cl.lqr(A, B, Q, R=0.1)
        solution_high_R = cl.lqr(A, B, Q, R=10.0)
        assert np.linalg.norm(solution_high_R.feedback_gain) < np.linalg.norm(
            solution_low_R.feedback_gain
        ), "Higher R should result in smaller gains"

    def test_lqr_eigenvalues_count(self, sample_4d_system):
        """Verify correct number of eigenvalues returned"""
        A, B, Q = sample_4d_system
        R = 1.0
        solution = cl.lqr(A, B, Q, R)
        assert (
            len(solution.closed_loop_eigenvalues) == 4
        ), "Should have 4 eigenvalues for 4D system"


# Unit tests for get_control_params() function
class TestGetControlParams:
    def test_get_control_params_basic(self):
        """Verify control parameters are computed correctly"""
        params = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=10.0, control_weight=1.0
        )
        assert params.lookahead_distance == DEFAULT_VEHICLE_PARAMS.l_s
        assert params.lqr_gain.shape == (4,)
        assert params.disturbance_compensation_gain > 0

    def test_get_control_params_gain_shape(self):
        """Verify LQR gain has correct shape"""
        params = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, control_weight=1.0
        )
        assert params.lqr_gain.shape == (4,), "LQR gain should have 4 elements"

    def test_get_control_params_different_velocities(self):
        """Control parameters should change with velocity"""
        params_slow = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=5.0, control_weight=1.0
        )
        params_fast = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=30.0, control_weight=1.0
        )
        # Gains should be different for different velocities
        assert not np.allclose(
            params_slow.lqr_gain, params_fast.lqr_gain
        ), "Gains should differ for different velocities"
        # Disturbance compensation should also differ
        assert params_slow.disturbance_compensation_gain != pytest.approx(
            params_fast.disturbance_compensation_gain
        ), "Disturbance compensation should differ for different velocities"

    def test_get_control_params_different_control_weights(self):
        """Higher control_weight should result in smaller gains (less aggressive)"""
        params_low_weight = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, control_weight=0.1
        )
        params_high_weight = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, control_weight=100.0
        )
        assert np.linalg.norm(params_high_weight.lqr_gain) < np.linalg.norm(
            params_low_weight.lqr_gain
        ), "Higher control_weight should result in smaller gains"

    def test_get_control_params_disturbance_compensation_positive(self):
        """Disturbance compensation should be positive for typical parameters"""
        params = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, control_weight=1.0
        )
        assert (
            params.disturbance_compensation_gain > 0
        ), "Disturbance compensation should be positive"

    def test_get_control_params_lookahead_distance(self):
        """Lookahead distance should match vehicle parameters"""
        params = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, control_weight=1.0
        )
        assert params.lookahead_distance == pytest.approx(
            DEFAULT_VEHICLE_PARAMS.l_s
        ), "Lookahead distance should match vehicle params"
