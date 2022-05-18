import numpy as np
import pytest

import behavior_generation_lecture_python.lateral_control_riccati.lateral_control_riccati as cl
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
