import numpy as np
import pytest

from behavior_generation_lecture_python.utils import generate_reference_curve


def test_straight_line():
    x_input = [0, 1, 2, 3]
    y_input = [0, 1, 2, 3]
    curve = generate_reference_curve.generate_reference_curve(x_input, y_input, 1.0)
    assert np.allclose(curve["x"], curve["y"])
    assert curve["s"][2] == pytest.approx(2.0)
    assert np.allclose(curve["kappa"], np.array([0.0] * len(curve["kappa"])))
    assert np.allclose(curve["theta"], np.array([np.pi / 4.0] * len(curve["theta"])))
