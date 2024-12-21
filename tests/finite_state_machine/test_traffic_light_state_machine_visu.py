

from behavior_generation_lecture_python.finite_state_machine import (
    traffic_light_state_machine_with_visu,
)


def test_traffic_light_state_machine_with_visu():
    tl = traffic_light_state_machine_with_visu.TrafficLightStateMachine(
        start_mainloop=False
    )
    tl.button_press(None)
