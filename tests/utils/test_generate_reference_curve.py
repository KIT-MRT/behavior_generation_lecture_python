import numpy as np
import pytest

from behavior_generation_lecture_python.utils import generate_reference_curve


def test_straight_line():
    x_input = np.array([0, 1, 2, 3])
    y_input = np.array([0, 1, 2, 3])
    curve = generate_reference_curve.generate_reference_curve(x_input, y_input, 1.0)
    assert np.allclose(curve.x, curve.y)
    assert curve.arc_length[2] == pytest.approx(2.0)
    assert np.allclose(curve.curvature, np.array([0.0] * len(curve.curvature)))
    assert np.allclose(curve.heading, np.array([np.pi / 4.0] * len(curve.heading)))
