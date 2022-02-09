import math
import numpy as np

from typing import Optional, Any, Dict, Set, List, Union, Tuple

SIMPLE_MDP_DICT = {
    "states": [1, 2],
    "actions": ["A", "B"],
    "initial_state": 1,
    "terminal_states": [2],
    "transition_probabilities": {
        (1, "A"): [(0.2, 1), (0.8, 2)],
        (1, "B"): [(0.5, 1), (0.5, 2)],
        (2, "A"): [(1.0, 1)],
        (2, "B"): [(0.3, 1), (0.7, 2)],
    },
    "reward": {1: -0.1, 2: -0.5},
}

GRID_MDP_DICT = {
    "grid": [
        [-0.04, -0.04, -0.04, +1],
        [-0.04, None, -0.04, -1],
        [-0.04, -0.04, -0.04, -0.04],
    ],
    "initial_state": (1, 0),
    "terminal_states": {(3, 2), (3, 1)},
    "transition_probabilities_per_action": {
        (0, 1): [(0.8, (0, 1)), (0.1, (1, 0)), (0.1, (-1, 0))],
        (0, -1): [(0.8, (0, -1)), (0.1, (1, 0)), (0.1, (-1, 0))],
        (1, 0): [(0.8, (1, 0)), (0.1, (0, 1)), (0.1, (0, -1))],
        (-1, 0): [(0.8, (-1, 0)), (0.1, (0, 1)), (0.1, (0, -1))],
    },
}

LC_LEFT_ACTION, STAY_IN_LANE_ACTION, LC_RIGHT_ACTION = (1, 1), (1, 0), (1, -1)

HIGHWAY_MDP_DICT = {
    "grid": [
        [0, -1, -1, -1, -1, -1, -1, -1, -1, -50],
        [0, -2, -2, -2, -2, -2, -2, -2, -2, -50],
        [0, -3, -3, -3, -3, -3, -3, -3, -3, -50],
        [None, None, None, None, None, None, -2, -2, -2, 0],
    ],
    "initial_state": (0, 2),
    "terminal_states": {(9, 3), (9, 1), (9, 2), (9, 0)},
    "transition_probabilities_per_action": {
        STAY_IN_LANE_ACTION: [(1.0, STAY_IN_LANE_ACTION)],
        LC_LEFT_ACTION: [(0.5, LC_LEFT_ACTION), (0.5, STAY_IN_LANE_ACTION)],
        LC_RIGHT_ACTION: [(0.75, LC_RIGHT_ACTION), (0.25, STAY_IN_LANE_ACTION)],
    },
    "restrict_actions_to_available_states": True,
}


class MDP:
    def __init__(
        self,
        states: Set[Any],
        actions: Set[Any],
        initial_state: Any,
        terminal_states: Set[Any],
        transition_probabilities: Dict[Tuple[Any, Any], List[Tuple[float, Any]]],
        reward: Dict[Any, float],
    ) -> None:
        """
        A Markov decision process.

        :param states: Set of states.
        :param actions: Set of actions.
        :param initial_state: Initial state.
        :param terminal_states: Set of terminal states.
        :param transition_probabilities: Dictionary of transition probabilities,
            mapping from tuple (state, action) to list of tuples (probability, next state).
        :param reward: Dictionary of rewards per state, mapping from state to reward.
        """
        self.states = states

        self.actions = actions

        assert initial_state in self.states
        self.initial_state = initial_state

        for terminal_state in terminal_states:
            assert (
                terminal_state in self.states
            ), f"The terminal state {terminal_state} is not in states {states}"
        self.terminal_states = terminal_states

        for state in self.states:
            for action in self.actions:
                if (state, action) not in transition_probabilities:
                    continue
                total_prob = 0
                for prob, next_state in transition_probabilities[(state, action)]:
                    assert (
                        next_state in self.states
                    ), f"next_state={next_state} is not in states={states}"
                    total_prob += prob
                assert math.isclose(total_prob, 1), "Probabilities must add to one"
        self.transition_probabilities = transition_probabilities

        assert set(reward.keys()) == set(self.states)
        for state in self.states:
            assert reward[state] is not None
        self.reward = reward

    def get_states(self) -> Set[Any]:
        """Get the set of states."""
        return self.states

    def get_actions(self, state) -> Set[Any]:
        """Get the set of actions available in a certain state, returns [None] for terminal states."""
        if self.is_terminal(state):
            return {None}
        return set(
            [a for a in self.actions if (state, a) in self.transition_probabilities]
        )

    def get_reward(self, state) -> float:
        """Get the reward for a specific state."""
        return self.reward[state]

    def is_terminal(self, state) -> bool:
        """Return whether a state is a terminal state."""
        return state in self.terminal_states

    def get_transitions_with_probabilities(
        self, state, action
    ) -> List[Tuple[float, Any]]:
        """Get the list of transitions with their probability, returns [(0.0, state)] for terminal states."""
        if action is None or self.is_terminal(state):
            return [(0.0, state)]
        return self.transition_probabilities[(state, action)]


class GridMDP(MDP):
    def __init__(
        self,
        grid: List[List[Union[float, None]]],
        initial_state: Tuple[int, int],
        terminal_states: Set[Tuple[int, int]],
        transition_probabilities_per_action: Dict[
            Tuple[int, int], List[Tuple[float, Tuple[int, int]]]
        ],
        restrict_actions_to_available_states: Optional[bool] = False,
    ) -> None:
        """
        A Markov decision process on a grid.

        :param grid: List of lists, containing the rewards of the grid states or None.
        :param initial_state: Initial state in the grid.
        :param terminal_states: Set of terminal states in the grid.
        :param transition_probabilities_per_action: Dictionary of transition probabilities per action,
            mapping from action to list of tuples (probability, next state).
        :param restrict_actions_to_available_states: Whether to restrict actions to those that result in valid
            next states.
        """
        states = set()
        reward = {}
        grid = grid.copy()
        grid.reverse()  # y-axis pointing upwards
        rows = len(grid)
        cols = len(grid[0])
        self.grid = grid
        for x in range(cols):
            for y in range(rows):
                if grid[y][x] is not None:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]

        transition_probabilities = {}
        for state in states:
            for action in transition_probabilities_per_action.keys():
                transition_probability_list = self._generate_transition_probability_list(
                    state=state,
                    action=action,
                    restrict_actions_to_available_states=restrict_actions_to_available_states,
                    states=states,
                    transition_probabilities_per_action=transition_probabilities_per_action,
                    next_state_fn=self._next_state_deterministic,
                )
                if len(transition_probability_list) > 0:
                    transition_probabilities[
                        (state, action)
                    ] = transition_probability_list

        super().__init__(
            states=states,
            actions=set(transition_probabilities_per_action.keys()),
            initial_state=initial_state,
            terminal_states=terminal_states,
            transition_probabilities=transition_probabilities,
            reward=reward,
        )

    @staticmethod
    def _generate_transition_probability_list(
        state,
        action,
        restrict_actions_to_available_states,
        states,
        transition_probabilities_per_action,
        next_state_fn,
    ):
        """Generate the transition probability list of the grid."""
        transition_probability_list = []
        none_in_next_states = False
        for (
            probability,
            deterministic_action,
        ) in transition_probabilities_per_action[action]:
            next_state = next_state_fn(
                state,
                deterministic_action,
                states,
                output_none_if_non_existing_state=restrict_actions_to_available_states,
            )
            if next_state is None:
                none_in_next_states = True
                break
            transition_probability_list.append((probability, next_state))

        if not none_in_next_states:
            return transition_probability_list

        return []

    @staticmethod
    def _next_state_deterministic(
        state, action, states, output_none_if_non_existing_state=False
    ):
        """Output the next state given the action in a deterministic setting.
        Output none if next state not existing in case output_none_if_non_existing_state is True."""
        next_state_candidate = tuple(np.array(state) + np.array(action))
        if next_state_candidate in states:
            return next_state_candidate
        if output_none_if_non_existing_state:
            return None
        return state


def expected_utility_of_action(
    mdp: MDP, state: Any, action: Any, utility_of_states: Dict[Any, float]
) -> float:
    """
    Compute the expected utility of taking an action in a state.

    :param mdp: The underlying MDP.
    :param state: The start state.
    :param action: The action to be taken.
    :param utility_of_states: The dictionary containing the utility (estimate) of all states.
    :return: Expected utility
    """
    return sum(
        p * utility_of_states[next_state]
        for (p, next_state) in mdp.get_transitions_with_probabilities(
            state=state, action=action
        )
    )


def derive_policy(mdp: MDP, utility_of_states: Dict[Any, float]) -> Dict[Any, Any]:
    """
    Compute the best policy for an MDP given the utility of the states.

    :param mdp: The underlying MDP.
    :param utility_of_states: The dictionary containing the utility (estimate) of all states.
    :return: Policy, i.e. mapping from state to action.
    """
    pi = {}
    for state in mdp.get_states():
        pi[state] = max(
            mdp.get_actions(state),
            key=lambda action: expected_utility_of_action(
                mdp=mdp, state=state, action=action, utility_of_states=utility_of_states
            ),
        )
    return pi


def value_iteration(
    mdp: MDP,
    epsilon: float,
    max_iterations: int,
    return_history: Optional[bool] = False,
) -> Union[Dict[Any, float], List[Dict[Any, float]]]:
    """
    Derive a utility estimate by means of value iteration.

    :param mdp: The underlying MDP.
    :param epsilon: Termination criterion:
        if maximum difference in utility update is below epsilon, the iteration is terminated.
    :param max_iterations: Maximum number of iterations, if exceeded, RuntimeError is raised.
    :param return_history: Whether to return the whole history of utilities instead of just the final estimate.
    :return: The final utility estimate, if return_history is false.
        The history of utility estimates as list, if return_history is true.
    """
    utility = {state: 0 for state in mdp.get_states()}
    utility_history = []
    for _ in range(max_iterations):
        utility_old = utility.copy()
        max_delta = 0
        for state in mdp.get_states():
            utility[state] = mdp.get_reward(state) + max(
                expected_utility_of_action(
                    mdp, state=state, action=action, utility_of_states=utility_old
                )
                for action in mdp.get_actions(state)
            )
            max_delta = max(max_delta, abs(utility[state] - utility_old[state]))
        if return_history:
            utility_history.append(utility.copy())
        if max_delta < epsilon:
            if return_history:
                return utility_history
            return utility
    raise RuntimeError(f"Did not converge in {max_iterations} iterations")
