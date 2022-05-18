import pytest

from behavior_generation_lecture_python.finite_state_machine import (
    traffic_light_state_machine,
)


def test_abstract_state():
    with pytest.raises(TypeError):
        traffic_light_state_machine.State()


def test_state_machine_init():
    class TestState(traffic_light_state_machine.State):
        transition_map = {}

        def run(self):
            print("bla")

    sm = traffic_light_state_machine.StateMachine(TestState())
    with pytest.warns(UserWarning):
        sm.trigger("does_not_exist_event")


def test_traffic_light_state_machine():
    tlsm = traffic_light_state_machine.StateMachine(
        initial_state=traffic_light_state_machine.Go()
    )

    with pytest.warns(UserWarning):
        assert "Go" == tlsm.trigger(
            traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
        )

    assert "PrepareToStop" == tlsm.trigger(
        traffic_light_state_machine.BUTTON_PRESSED_EVENT
    )

    assert "Stop" == tlsm.trigger(
        traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
    )

    with pytest.warns(UserWarning):
        assert "Stop" == tlsm.trigger(traffic_light_state_machine.BUTTON_PRESSED_EVENT)

    assert "PrepareToStart" == tlsm.trigger(
        traffic_light_state_machine.PEDESTRIAN_TIMER_ELAPSED_EVENT
    )

    assert "Go" == tlsm.trigger(
        traffic_light_state_machine.LIGHT_CHANGE_TIMER_ELAPSED_EVENT
    )
