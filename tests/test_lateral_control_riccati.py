import numpy as np
import pytest

import behavior_generation_lecture_python.lateral_control_riccati.lateral_control_riccati as cl
import behavior_generation_lecture_python.lateral_control_riccati.riccati_controller as con
import behavior_generation_lecture_python.utils.generate_reference_curve as ref
from behavior_generation_lecture_python.utils.projection import project2curve
from behavior_generation_lecture_python.vehicle_models.vehicle_parameters import (
    DEFAULT_VEHICLE_PARAMS,
)


@pytest.mark.parametrize("test_r,error_factor", [(10000, 1), (10, 0.5)])
def test_lateral_control_riccati(test_r, error_factor):
    radius = 500
    vars_0 = [0.0, -radius, 0.0, 0.0, 0.0]
    v_0 = 33.0

    curve = ref.generate_reference_curve(
        [0, radius, 0, -radius, 0], [-radius, 0, radius, 0, radius], 10.0
    )
    ti = np.arange(0, 40, 0.1)

    model = cl.LateralControlRiccati(
        initial_condition=vars_0,
        curve=curve,
        vehicle_params=DEFAULT_VEHICLE_PARAMS,
        initial_velocity=v_0,
        r=test_r,
    )
    sol = model.simulate(ti, v=v_0, t_step=0.1)

    errors = []
    for state in sol:
        _, _, _, d, _, _ = project2curve(
            s_c=curve["s"],
            x_c=curve["x"],
            y_c=curve["y"],
            theta_c=curve["theta"],
            kappa_c=curve["kappa"],
            x=state[0],
            y=state[1],
        )
        errors.append(abs(d))

    assert np.sum(errors) > len(sol) * 0.1 * error_factor
    assert np.sum(errors) < len(sol) * 0.2 * error_factor


# Unit tests for riccati_controller.feedback_law()
class TestFeedbackLaw:
    def test_feedback_law_zero_error(self):
        """Zero steering when on reference with no curvature and no dynamics"""
        k_lqr = np.array([1.0, 1.0, 1.0, 1.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=1.0, e_l=0, e_psi=0, kappa_r=0, beta=0, r=0
        )
        assert delta == pytest.approx(0)

    def test_feedback_law_lateral_error_positive(self):
        """Negative steering to correct positive lateral error (left of reference)"""
        k_lqr = np.array([1.0, 0.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=1.0, e_psi=0, kappa_r=0, beta=0, r=0
        )
        assert delta < 0, "Should steer right (negative) to correct left error"

    def test_feedback_law_lateral_error_negative(self):
        """Positive steering to correct negative lateral error (right of reference)"""
        k_lqr = np.array([1.0, 0.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=-1.0, e_psi=0, kappa_r=0, beta=0, r=0
        )
        assert delta > 0, "Should steer left (positive) to correct right error"

    def test_feedback_law_heading_error_positive(self):
        """Steering correction for positive heading error"""
        k_lqr = np.array([0.0, 1.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=0, e_psi=0.1, kappa_r=0, beta=0, r=0
        )
        assert delta < 0, "Should steer right to correct heading pointing left"

    def test_feedback_law_heading_error_negative(self):
        """Steering correction for negative heading error"""
        k_lqr = np.array([0.0, 1.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=0, e_psi=-0.1, kappa_r=0, beta=0, r=0
        )
        assert delta > 0, "Should steer left to correct heading pointing right"

    def test_feedback_law_curvature_feedforward_positive(self):
        """Positive curvature should add positive steering component"""
        k_lqr = np.array([0.0, 0.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=1.0, e_l=0, e_psi=0, kappa_r=0.1, beta=0, r=0
        )
        assert delta > 0, "Positive curvature should cause positive steering"

    def test_feedback_law_curvature_feedforward_negative(self):
        """Negative curvature should add negative steering component"""
        k_lqr = np.array([0.0, 0.0, 0.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=1.0, e_l=0, e_psi=0, kappa_r=-0.1, beta=0, r=0
        )
        assert delta < 0, "Negative curvature should cause negative steering"

    def test_feedback_law_beta_correction(self):
        """Sideslip angle (beta) should be corrected"""
        k_lqr = np.array([0.0, 0.0, 1.0, 0.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=0, e_psi=0, kappa_r=0, beta=0.1, r=0
        )
        assert delta < 0, "Positive beta should cause negative steering correction"

    def test_feedback_law_yaw_rate_correction(self):
        """Yaw rate (r) should be corrected"""
        k_lqr = np.array([0.0, 0.0, 0.0, 1.0])
        delta = con.feedback_law(
            k_lqr=k_lqr, k_dist_comp=0.0, e_l=0, e_psi=0, kappa_r=0, beta=0, r=0.1
        )
        assert delta < 0, "Positive yaw rate should cause negative steering correction"


# Unit tests for lqr() function
# Note: The lqr() function is specifically designed for 4-state systems
class TestLQR:
    @pytest.fixture
    def sample_4d_system(self):
        """A sample 4D system similar to vehicle lateral control"""
        A = np.array(
            [[0, 1, 1, 0], [0, 0, 0, 1], [0, 0, -1, 0.5], [0, 0, 0.5, -1]]
        )
        b = np.array([[0], [0], [1], [0.5]])
        Q = np.eye(4)
        return A, b, Q

    def test_lqr_4d_system_stability(self, sample_4d_system):
        """Test LQR on 4D system - closed loop should be stable"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        # Verify closed-loop eigenvalues are stable (negative real parts)
        assert all(
            np.real(eig_vals) < 0
        ), "Closed-loop system should have stable eigenvalues"

    def test_lqr_gain_shape(self, sample_4d_system):
        """Verify K has correct shape for 4D system"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        assert K.shape == (4,), "K should have 4 elements for 4D system"

    def test_lqr_riccati_solution_shape(self, sample_4d_system):
        """Verify X (Riccati solution) has correct shape"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        assert X.shape == (4, 4), "X should be 4x4 for 4D system"

    def test_lqr_riccati_solution_symmetric(self, sample_4d_system):
        """Verify X (Riccati solution) is symmetric"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        assert np.allclose(X, X.T), "Riccati solution should be symmetric"

    def test_lqr_riccati_solution_positive_semidefinite(self, sample_4d_system):
        """Verify X (Riccati solution) is positive semi-definite"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        eigenvalues_X = np.linalg.eigvals(X)
        assert all(
            eigenvalues_X >= -1e-10
        ), "Riccati solution should be positive semi-definite"

    def test_lqr_higher_control_cost_smaller_gains(self, sample_4d_system):
        """Higher control cost r should result in smaller gains"""
        A, b, Q = sample_4d_system
        K_low_r, _, _ = cl.lqr(A, b, Q, r=0.1)
        K_high_r, _, _ = cl.lqr(A, b, Q, r=10.0)
        assert np.linalg.norm(K_high_r) < np.linalg.norm(
            K_low_r
        ), "Higher r should result in smaller gains"

    def test_lqr_eigenvalues_count(self, sample_4d_system):
        """Verify correct number of eigenvalues returned"""
        A, b, Q = sample_4d_system
        r = 1.0
        K, X, eig_vals = cl.lqr(A, b, Q, r)
        assert len(eig_vals) == 4, "Should have 4 eigenvalues for 4D system"


# Unit tests for get_control_params() function
class TestGetControlParams:
    def test_get_control_params_basic(self):
        """Verify control parameters are computed correctly"""
        params = cl.get_control_params(DEFAULT_VEHICLE_PARAMS, velocity=10.0, r=1.0)
        assert params.l_s == DEFAULT_VEHICLE_PARAMS.l_s
        assert params.k_lqr.shape == (4,)
        assert params.k_dist_comp > 0

    def test_get_control_params_gain_shape(self):
        """Verify LQR gain has correct shape"""
        params = cl.get_control_params(DEFAULT_VEHICLE_PARAMS, velocity=20.0, r=1.0)
        assert params.k_lqr.shape == (4,), "LQR gain should have 4 elements"

    def test_get_control_params_different_velocities(self):
        """Control parameters should change with velocity"""
        params_slow = cl.get_control_params(DEFAULT_VEHICLE_PARAMS, velocity=5.0, r=1.0)
        params_fast = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=30.0, r=1.0
        )
        # Gains should be different for different velocities
        assert not np.allclose(
            params_slow.k_lqr, params_fast.k_lqr
        ), "Gains should differ for different velocities"
        # Disturbance compensation should also differ
        assert params_slow.k_dist_comp != pytest.approx(
            params_fast.k_dist_comp
        ), "Disturbance compensation should differ for different velocities"

    def test_get_control_params_different_r_values(self):
        """Higher r should result in smaller gains (less aggressive control)"""
        params_low_r = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, r=0.1
        )
        params_high_r = cl.get_control_params(
            DEFAULT_VEHICLE_PARAMS, velocity=20.0, r=100.0
        )
        assert np.linalg.norm(params_high_r.k_lqr) < np.linalg.norm(
            params_low_r.k_lqr
        ), "Higher r should result in smaller gains"

    def test_get_control_params_disturbance_compensation_positive(self):
        """Disturbance compensation should be positive for typical parameters"""
        params = cl.get_control_params(DEFAULT_VEHICLE_PARAMS, velocity=20.0, r=1.0)
        assert params.k_dist_comp > 0, "Disturbance compensation should be positive"

    def test_get_control_params_lookahead_distance(self):
        """Lookahead distance should match vehicle parameters"""
        params = cl.get_control_params(DEFAULT_VEHICLE_PARAMS, velocity=20.0, r=1.0)
        assert params.l_s == pytest.approx(
            DEFAULT_VEHICLE_PARAMS.l_s
        ), "Lookahead distance should match vehicle params"
