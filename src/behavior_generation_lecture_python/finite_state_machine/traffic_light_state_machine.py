import warnings
from abc import ABC, abstractmethod


class StateMachine:
    """Finite state machine"""

    def __init__(self, initial_state):
        self.current_state = initial_state
        self.current_state.run()

    def trigger(self, event):
        """Trigger an event and run the subsequent state

        Args:
            event: event to be triggered

        Returns:
            Name of the current state
        """
        self.current_state = self.current_state.transition(event)
        self.current_state.run()
        return self.get_state_name()

    def get_state_name(self):
        """Return the name of the state as string

        Returns:
            Name of the state
        """
        return type(self.current_state).__name__


class State(ABC):
    """Abstract base class for a state"""

    transition_map = None

    @abstractmethod
    def run(self):
        """Abstract run method of the state"""

    def transition(self, event):
        """Transition to the next state based on the event

        Args:
            event: Event on which the transition is based

        Returns:
            next state
        """
        print(f"Event '{event}' called")
        if event in self.transition_map:
            print(self.transition_map[event])
            return eval(self.transition_map[event])()
        warnings.warn(
            f"No transition for '{event}' defined in state '{type(self).__name__}'"
        )
        return self


BUTTON_PRESSED_EVENT = "button_pressed_event"
LIGHT_CHANGE_TIMER_ELAPSED_EVENT = "light_change_timer_elapsed_event"
PEDESTRIAN_TIMER_ELAPSED_EVENT = "pedestrian_green_timer_elapsed_event"


class Go(State):
    """Vehicles can go"""

    transition_map = {BUTTON_PRESSED_EVENT: "PrepareToStop"}

    def run(self):
        print("vehicles: grn, pedestrians: red")


class PrepareToStop(State):
    """Vehicles shall prepare to stop"""

    transition_map = {LIGHT_CHANGE_TIMER_ELAPSED_EVENT: "Stop"}

    def run(self):
        print("vehicles: yel, pedestrians: yel")


class Stop(State):
    """Vehicles shall stop"""

    transition_map = {PEDESTRIAN_TIMER_ELAPSED_EVENT: "PrepareToStart"}

    def run(self):
        print("vehicles: red, pedestrians: grn")


class PrepareToStart(State):
    """Vehicles shall prepare to go"""

    transition_map = {LIGHT_CHANGE_TIMER_ELAPSED_EVENT: "Go"}

    def run(self):
        print("vehicles: yel, pedestrians: yel")
