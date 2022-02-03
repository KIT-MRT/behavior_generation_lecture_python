import numpy as np
import pytest

from utils import projection


class SampleCurve:
    def __init__(self):
        self.curve_points_x = [0, 1, 2, 3]
        self.curve_points_y = [0, 0, 1, 3]
        self.curve_points_s = [0, 1, 1 + np.sqrt(2), 1 + np.sqrt(1 + 2 ** 2)]
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
        projected_point = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    x_p, y_p, s_p, d, theta_p, kappa_p = projected_point
    assert x_p == -1
    assert y_p == 0
    assert s_p == -1
    assert d == pytest.approx(0)
    assert kappa_p == -10
    # theta is not meaningful here

    target_point = [4, 3]
    with pytest.warns(UserWarning, match="Extrapolating over end of reference curve!"):
        projected_point = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    x_p, y_p, s_p, d, theta_p, kappa_p = projected_point
    assert x_p == pytest.approx(3.2, abs=0.1)
    assert y_p == pytest.approx(3.4, abs=0.1)
    assert s_p == pytest.approx(3.4, abs=0.1)
    assert d == pytest.approx(-0.9, abs=0.1)
    assert theta_p == pytest.approx(np.arctan(2))
    assert kappa_p == -10

    target_point = [1, 1]
    with pytest.warns(None) as recorded_warnings:
        projected_point = projection.project2curve(
            s_c=sc.curve_points_s,
            x_c=sc.curve_points_x,
            y_c=sc.curve_points_y,
            theta_c=sc.curve_points_theta,
            kappa_c=sc.curve_points_kappa_fantasy,
            x=target_point[0],
            y=target_point[1],
        )
    assert len(recorded_warnings) == 0
    x_p, y_p, s_p, d, theta_p, kappa_p = projected_point
    # todo: assertions
