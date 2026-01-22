import numpy as np
import pytest
import warnings

from behavior_generation_lecture_python.utils import projection


class SampleCurve:
    def __init__(self):
        self.curve_points_x = [0, 1, 2, 3]
        self.curve_points_y = [0, 0, 1, 3]
        self.curve_points_s = [0, 1, 1 + np.sqrt(2), 1 + np.sqrt(1 + 2**2)]
        self.curve_points_theta = [0, np.arctan(1), np.arctan(2), np.arctan(2)]
        self.curve_points_kappa_fantasy = [-10, -10, -10, -10]


def test_pseudo_projection():
    sc = SampleCurve()
    target_point = [1, 1]
    projected_point = projection.pseudo_projection(
        start_index=1,
        x=target_point[0],
        y=target_point[1],
        x_c=sc.curve_points_x,
        y_c=sc.curve_points_y,
        theta_c=sc.curve_points_theta,
    )
    px, lambda_, sign = projected_point
    # todo: assertions


def test_project2curve():
    sc = SampleCurve()
    target_point = [-1, 0]
    with pytest.warns(
        UserWarning, match="Extrapolating over start of reference curve!"
    ):
        result = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    assert result.x == -1
    assert result.y == 0
    assert result.arc_length == -1
    assert result.lateral_error == pytest.approx(0)
    assert result.curvature == -10
    # heading is not meaningful here

    target_point = [4, 3]
    with pytest.warns(UserWarning, match="Extrapolating over end of reference curve!"):
        result = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    assert result.x == pytest.approx(3.2, abs=0.1)
    assert result.y == pytest.approx(3.4, abs=0.1)
    assert result.arc_length == pytest.approx(3.4, abs=0.1)
    assert result.lateral_error == pytest.approx(-0.9, abs=0.1)
    assert result.heading == pytest.approx(np.arctan(2))
    assert result.curvature == -10

    target_point = [1, 1]
    with warnings.catch_warnings():
        result = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    # todo: assertions
