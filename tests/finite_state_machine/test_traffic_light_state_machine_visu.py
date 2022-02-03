import pytest
import os

from finite_state_machine import traffic_light_state_machine_with_visu


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="Display required")
def test_traffic_light_state_machine_with_visu():
    tl = traffic_light_state_machine_with_visu.TrafficLightStateMachine(
        start_mainloop=False
    )
    tl.button_press(None)
