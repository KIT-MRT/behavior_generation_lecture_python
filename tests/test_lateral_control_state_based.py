import numpy as np

import behavior_generation_lecture_python.lateral_control_state_based.lateral_control_state_based as cl
import behavior_generation_lecture_python.utils.generate_reference_curve as ref
from behavior_generation_lecture_python.utils.projection import project2curve


def test_lateral_control_state_based():

    radius = 20
    vars_0 = [0.1, -radius, 0.0]
    curve = ref.generate_reference_curve(
        [0, radius, 0, -radius, 0], [-radius, 0, radius, 0, radius], 1.0
    )
    ti = np.arange(0, 100, 0.1)
    model = cl.LateralControlStateBased(vars_0, curve)
    sol = model.simulate(ti, v=1)

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

    assert np.sum(errors) < len(sol) * 0.01
