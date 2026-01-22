import math

import pytest

from behavior_generation_lecture_python.lateral_control_state_based import (
    state_based_controller,
)


def test_feedback_law():
    assert state_based_controller.feedback_law(
        d=0, psi=0, theta_r=0, kappa_r=0
    ) == pytest.approx(0), "zero steering for going straight"

    assert (
        state_based_controller.feedback_law(d=0.1, psi=0, theta_r=0, kappa_r=0) < 0
    ), "neg. steering (to the right) if left of reference curve"

    assert (
        state_based_controller.feedback_law(d=-0.1, psi=0, theta_r=0, kappa_r=0) > 0
    ), "pos. steering (to the left) if right of reference curve"

    assert (
        state_based_controller.feedback_law(d=0, psi=0, theta_r=0, kappa_r=0.1) > 0
    ), "positive steering (to the left) if on reference curve and ref curve has positive curvature"

    assert (
        state_based_controller.feedback_law(d=0, psi=0, theta_r=0, kappa_r=-0.1) < 0
    ), "negative steering (to the right) if on reference curve and ref curve has negative curvature"

    assert (
        state_based_controller.feedback_law(d=0, psi=0.1, theta_r=0.2, kappa_r=0) > 0
    ), "positive steering (to the left) if reference curve heads further left"

    assert (
        state_based_controller.feedback_law(d=0, psi=-0.1, theta_r=-0.2, kappa_r=0) < 0
    ), "negative steering (to the right) if reference curve heads further right"


def test_feedback_law_angle_wrapping():
    """Test behavior when angles are near +/- pi boundaries"""
    # psi near +pi, theta_r near -pi (they are actually close due to wrapping)
    # The normalized difference should be small, so steering should be small
    delta = state_based_controller.feedback_law(d=0, psi=3.0, theta_r=-3.0, kappa_r=0)
    # The angle difference after normalization: 3.0 - (-3.0) = 6.0
    # but normalized: 6.0 wraps to about -0.28 radians
    # So steering should be positive (steer left) but small
    assert abs(delta) < 1.0, "Angle wrapping should result in reasonable steering"

    # Test exact pi boundary
    delta_at_pi = state_based_controller.feedback_law(
        d=0, psi=math.pi - 0.1, theta_r=-math.pi + 0.1, kappa_r=0
    )
    # These angles are close (differ by ~0.2 radians across the pi boundary)
    assert abs(delta_at_pi) < 0.6, "Steering near pi boundary should be reasonable"


def test_feedback_law_straight_line():
    """Test with zero curvature (straight reference)"""
    # Heading error to the left, should steer right
    delta = state_based_controller.feedback_law(d=0, psi=0.1, theta_r=0, kappa_r=0)
    assert delta < 0, "Should steer right to correct left heading error"

    # Heading error to the right, should steer left
    delta = state_based_controller.feedback_law(d=0, psi=-0.1, theta_r=0, kappa_r=0)
    assert delta > 0, "Should steer left to correct right heading error"


def test_feedback_law_combined_errors():
    """Test with multiple error sources combined"""
    # Left of reference (d > 0) and heading left (psi > theta_r)
    # Both errors suggest steering right (negative)
    delta = state_based_controller.feedback_law(d=1.0, psi=0.2, theta_r=0, kappa_r=0)
    assert delta < 0, "Combined errors should reinforce steering direction"

    # Right of reference (d < 0) and heading right (psi < theta_r)
    # Both errors suggest steering left (positive)
    delta = state_based_controller.feedback_law(d=-1.0, psi=-0.2, theta_r=0, kappa_r=0)
    assert delta > 0, "Combined errors should reinforce steering direction"


def test_feedback_law_opposing_errors():
    """Test with opposing error sources"""
    # Left of reference (d > 0) but heading right (psi < theta_r)
    # Errors partially cancel
    delta_opposing = state_based_controller.feedback_law(
        d=0.5, psi=-0.1, theta_r=0, kappa_r=0
    )
    delta_lateral_only = state_based_controller.feedback_law(
        d=0.5, psi=0, theta_r=0, kappa_r=0
    )
    # Opposing heading should reduce the steering magnitude
    assert abs(delta_opposing) < abs(
        delta_lateral_only
    ), "Opposing errors should reduce steering"


def test_feedback_law_large_lateral_error():
    """Test with large lateral errors"""
    delta_large = state_based_controller.feedback_law(
        d=10.0, psi=0, theta_r=0, kappa_r=0
    )
    delta_small = state_based_controller.feedback_law(
        d=0.1, psi=0, theta_r=0, kappa_r=0
    )
    assert abs(delta_large) > abs(
        delta_small
    ), "Larger lateral error should cause larger steering"
    # Steering should be bounded by arctan
    assert abs(delta_large) < math.pi / 2, "Steering should be bounded"


def test_feedback_law_large_curvature():
    """Test with large reference curvature"""
    delta_large_kappa = state_based_controller.feedback_law(
        d=0, psi=0, theta_r=0, kappa_r=1.0
    )
    delta_small_kappa = state_based_controller.feedback_law(
        d=0, psi=0, theta_r=0, kappa_r=0.01
    )
    assert abs(delta_large_kappa) > abs(
        delta_small_kappa
    ), "Larger curvature should cause larger steering"
