import numpy as np
import pytest

from utils.normalize_angle import normalize_angle


def test_normalize_angle():
    assert normalize_angle(3 * np.pi) == pytest.approx(-np.pi)
    assert normalize_angle(3.5 * np.pi) == pytest.approx(-0.5 * np.pi)
    assert normalize_angle(4.5 * np.pi) == pytest.approx(0.5 * np.pi)
    assert normalize_angle(0.5 * np.pi) == pytest.approx(0.5 * np.pi)
