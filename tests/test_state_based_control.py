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
