from behavior_generation_lecture_python.finite_state_machine import (
    traffic_light_state_machine,
)

if __name__ == "__main__":
    tlsm = traffic_light_state_machine.StateMachine(
        initial_state=traffic_light_state_machine.Go()
    )

    assert "Go" == tlsm.trigger(
        traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
    )

    assert "PrepareToStop" == tlsm.trigger(
        traffic_light_state_machine.BUTTON_PRESSED_EVENT
    )

    assert "Stop" == tlsm.trigger(
        traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
    )

    assert "Stop" == tlsm.trigger(traffic_light_state_machine.BUTTON_PRESSED_EVENT)

    assert "PrepareToStart" == tlsm.trigger(
        traffic_light_state_machine.PEDESTRIAN_TIMER_ELAPSED_EVENT
    )

    assert "Go" == tlsm.trigger(
        traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
    )
