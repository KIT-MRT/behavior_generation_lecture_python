import numpy as np

import behavior_generation_lecture_python.lateral_control_state_based.lateral_control_state_based as cl
import behavior_generation_lecture_python.utils.generate_reference_curve as ref
from behavior_generation_lecture_python.lateral_control_state_based.lateral_control_state_based import (
    KinematicVehicleState,
)


def test_lateral_control_state_based():
    radius = 20
    initial_state = KinematicVehicleState(x=0.1, y=-radius, heading=0.0)
    curve = ref.generate_reference_curve(
        np.array([0, radius, 0, -radius, 0]),
        np.array([-radius, 0, radius, 0, radius]),
        1.0,
    )
    time_vector = np.arange(0, 100, 0.1)
    model = cl.LateralControlStateBased(initial_state, curve)
    trajectory = model.simulate(time_vector, velocity=1)

    # trajectory is now a list of ControllerOutput dataclasses
    errors = [abs(output.lateral_error) for output in trajectory]

    assert np.sum(errors) < len(trajectory) * 0.01
