import math
import torch
import numpy as np
from copy import deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from behavior_generation_lecture_python.mdp.policy import CategorialPolicy

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
        """A Markov decision process.

        Args:
            states: Set of states.
            actions: Set of actions.
            initial_state: Initial state.
            terminal_states: Set of terminal states.
            transition_probabilities: Dictionary of transition
                probabilities, mapping from tuple (state, action) to
                list of tuples (probability, next state).
            reward: Dictionary of rewards per state, mapping from state
                to reward.
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

        assert set(reward.keys()) == set(
            self.states
        ), "Rewards must be defined for every state in the set of states"
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

    def sample_next_state(self, state, action) -> Any:
        """Randomly sample the next state given the current state and taken action."""
        if self.is_terminal(state):
            return ValueError("No next state for terminal states.")
        if action is None:
            return ValueError("Action must not be None.")
        prob_per_transition = self.get_transitions_with_probabilities(state, action)
        num_actions = len(prob_per_transition)
        choice = np.random.choice(
            num_actions, p=[ppa[0] for ppa in prob_per_transition]
        )
        return prob_per_transition[choice][1]

    def execute_action(self, state, action) -> Tuple[Any, float, bool]:
        """Executes the action in the current state and returns the new state, obtained reward and terminal flag."""
        new_state = self.sample_next_state(state=state, action=action)
        reward = self.get_reward(state=new_state)
        terminal = self.is_terminal(state=new_state)
        return new_state, reward, terminal


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
        """A Markov decision process on a grid.

        Args:
            grid: List of lists, containing the rewards of the grid
                states or None.
            initial_state: Initial state in the grid.
            terminal_states: Set of terminal states in the grid.
            transition_probabilities_per_action: Dictionary of
                transition probabilities per action, mapping from action
                to list of tuples (probability, next state).
            restrict_actions_to_available_states: Whether to restrict
                actions to those that result in valid next states.
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
                if transition_probability_list:
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
        Output None if next state not existing in case output_none_if_non_existing_state is True.
        """
        next_state_candidate = tuple(np.array(state) + np.array(action))
        if next_state_candidate in states:
            return next_state_candidate
        if output_none_if_non_existing_state:
            return None
        return state


def expected_utility_of_action(
    mdp: MDP, state: Any, action: Any, utility_of_states: Dict[Any, float]
) -> float:
    """Compute the expected utility of taking an action in a state.

    Args:
        mdp: The underlying MDP.
        state: The start state.
        action: The action to be taken.
        utility_of_states: The dictionary containing the utility
            (estimate) of all states.

    Returns:
        Expected utility
    """
    return sum(
        p * (mdp.get_reward(next_state) + utility_of_states[next_state])
        for (p, next_state) in mdp.get_transitions_with_probabilities(
            state=state, action=action
        )
    )


def derive_policy(mdp: MDP, utility_of_states: Dict[Any, float]) -> Dict[Any, Any]:
    """Compute the best policy for an MDP given the utility of the states.

    Args:
        mdp: The underlying MDP.
        utility_of_states: The dictionary containing the utility
            (estimate) of all states.

    Returns:
        Policy, i.e. mapping from state to action.
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
    """Derive a utility estimate by means of value iteration.

    Args:
        mdp: The underlying MDP.
        epsilon: Termination criterion: if maximum difference in utility
            update is below epsilon, the iteration is terminated.
        max_iterations: Maximum number of iterations, if exceeded,
            RuntimeError is raised.
        return_history: Whether to return the whole history of utilities
            instead of just the final estimate.

    Returns:
        The final utility estimate, if return_history is false. The
        history of utility estimates as list, if return_history is true.
    """
    utility = {state: 0 for state in mdp.get_states()}
    utility_history = [utility.copy()]
    for _ in range(max_iterations):
        utility_old = utility.copy()
        max_delta = 0
        for state in mdp.get_states():
            utility[state] = max(
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


def best_action_from_q_table(
    *, state: Any, available_actions: Set[Any], q_table: Dict[Tuple[Any, Any], float]
) -> Any:
    """Derive the best action from a Q table.

    Args:
        state: The state in which to take an action.
        available_actions: Set of available actions.
        q_table: The Q table, mapping from state-action pair to value estimate.

    Returns:
        The best action according to the Q table.
    """
    available_actions = list(available_actions)
    values = np.array([q_table[(state, action)] for action in available_actions])
    action = available_actions[np.argmax(values)]
    return action


def random_action(available_actions: Set[Any]) -> Any:
    """Derive a random action from the set of available actions.

    Args:
        available_actions: Set of available actions.

    Returns:
        A random action.
    """
    available_actions = list(available_actions)
    num_actions = len(available_actions)
    choice = np.random.choice(num_actions)
    return available_actions[choice]


def greedy_value_estimate_for_state(
    *, q_table: Dict[Tuple[Any, Any], float], state: Any
) -> float:
    """Compute the greedy (best possible) value estimate for a state from the Q table.

    Args:
        state: The state for which to estimate the value, when being greedy.
        q_table: The Q table, mapping from state-action pair to value estimate.

    Returns:
        The value based on the greedy estimate.
    """
    available_actions = [
        state_action[1] for state_action in q_table.keys() if state_action[0] == state
    ]
    return max([q_table[(state, action)] for action in available_actions])


def q_learning(
    *,
    mdp: MDP,
    alpha: float,
    epsilon: float,
    iterations: int,
    return_history: Optional[bool] = False,
) -> Dict[Tuple[Any, Any], float]:
    """Derive a value estimate for state-action pairs by means of Q learning.

    Args:
        mdp: The underlying MDP.
        alpha: Learning rate.
        epsilon: Exploration-exploitation threshold. A random action is taken with
            probability epsilon, the best action otherwise.
        iterations: Number of iterations.
        return_history: Whether to return the whole history of value estimates
            instead of just the final estimate.

    Returns:
        The final value estimate, if return_history is false. The
        history of value estimates as list, if return_history is true.
    """
    q_table = {}
    for state in mdp.get_states():
        for action in mdp.get_actions(state):
            q_table[(state, action)] = 0.0
    q_table_history = [q_table.copy()]
    state = mdp.initial_state

    np.random.seed(1337)

    for _ in range(iterations):

        # available actions:
        avail_actions = mdp.get_actions(state)

        # choose action (exploration-exploitation trade-off)
        rand = np.random.random()
        if rand < (1 - epsilon):
            chosen_action = best_action_from_q_table(
                state=state, available_actions=avail_actions, q_table=q_table
            )
        else:
            chosen_action = random_action(avail_actions)

        # interact with environment
        next_state = mdp.sample_next_state(state, chosen_action)

        # update Q table
        greedy_value_estimate_next_state = greedy_value_estimate_for_state(
            q_table=q_table, state=next_state
        )
        q_table[(state, chosen_action)] = (1 - alpha) * q_table[
            (state, chosen_action)
        ] + alpha * (mdp.get_reward(next_state) + greedy_value_estimate_next_state)

        if return_history:
            q_table_history.append(q_table.copy())

        if mdp.is_terminal(next_state):
            state = mdp.initial_state  # restart
        else:
            state = next_state  # continue

    if return_history:
        utility_history = []
        for q_tab in q_table_history:
            utility_history.append(
                {
                    state: greedy_value_estimate_for_state(q_table=q_tab, state=state)
                    for state in mdp.get_states()
                }
            )
        return utility_history

    return {
        state: greedy_value_estimate_for_state(q_table=q_table, state=state)
        for state in mdp.get_states()
    }


def policy_gradient(
    *,
    mdp: MDP,
    pol: CategorialPolicy,
    lr: float = 1e-2,
    iterations: int = 50,
    batch_size: int = 5000,
    return_history: bool = False,
    use_random_init_state: bool = False,
    verbose: bool = True,
) -> Union[List[CategorialPolicy], CategorialPolicy]:
    """Train a paramterized policy using vanilla policy gradient.

    Adapted from: https://github.com/openai/spinningup/blob/master/spinup/examples/pytorch/pg_math/1_simple_pg.py

    The MIT License (MIT)

    Copyright (c) 2018 OpenAI (http://openai.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

    Args:
        mdp: The underlying MDP.
        pol: The stochastic policy to be trained.
        lr: Learning rate.
        iterations: Number of iterations.
        batch_size: Number of samples generated for each policy update.
        return_history: Whether to return the whole history of value estimates
            instead of just the final estimate.
        use_random_init_state: bool, if the agent should be initialized randomly.
        verbose: bool, if traing progress should be printed.

    Returns:
        The final policy, if return_history is false. The
        history of policies as list, if return_history is true.
    """
    np.random.seed(1337)
    torch.manual_seed(1337)

    # add untrained model to model_checkpoints
    model_checkpoints = [deepcopy(pol)]

    # make optimizer
    optimizer = torch.optim.Adam(pol.net.parameters(), lr=lr)

    # get non-terminal states
    non_terminal_states = [state for state in mdp.states if not mdp.is_terminal(state)]

    # training loop
    for i in range(1, iterations + 1):

        # make some empty lists for logging.
        buffer = {
            "states": [],
            "actions": [],
            "weights": [],
            "ep_rets": [],
            "ep_lens": [],
        }

        # reset episode-specific variables
        if use_random_init_state:
            state = non_terminal_states[np.random.choice(len(non_terminal_states))]
        else:
            state = mdp.initial_state
        episode_rewards = []

        # collect experience by acting in the mdp
        while True:
            # save visited state
            buffer["states"].append(deepcopy(state))

            # call model to get next action
            action = pol.get_action(state=torch.as_tensor(state, dtype=torch.float32))

            # execute action in the environment
            state, reward, done = mdp.execute_action(state=state, action=action)

            # save action, reward
            buffer["actions"].append(action)
            episode_rewards.append(reward)

            if done:
                # if episode is over, record info about episode
                episode_return = sum(episode_rewards)
                episode_length = len(episode_rewards)
                buffer["ep_rets"].append(episode_return)
                buffer["ep_lens"].append(episode_length)
                # the weight for each logprob(a|s) is R(tau)
                buffer["weights"] += [episode_return] * episode_length

                # reset episode-specific variables
                if use_random_init_state:
                    state = non_terminal_states[
                        np.random.choice(len(non_terminal_states))
                    ]
                else:
                    state = mdp.initial_state
                episode_rewards = []

                # end experience loop if we have enough of it
                if len(buffer["states"]) > batch_size:
                    break

        # compute the loss
        logp = pol.get_log_prob(
            states=torch.as_tensor(buffer["states"], dtype=torch.float32),
            actions=torch.as_tensor(buffer["actions"], dtype=torch.int32),
        )
        batch_loss = -(
            logp * torch.as_tensor(buffer["weights"], dtype=torch.float32)
        ).mean()

        # take a single policy gradient update step
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # logging
        if verbose:
            print(
                "iteration: %3d;  return: %.3f;  episode_length: %.3f"
                % (i, np.mean(buffer["ep_rets"]), np.mean(buffer["ep_lens"]))
            )
        if return_history:
            model_checkpoints.append(deepcopy(pol))
    if return_history:
        return model_checkpoints
    return pol


def derive_deterministic_policy(mdp: MDP, pol: CategorialPolicy) -> Dict[Any, Any]:
    """Compute the best policy for an MDP given the stochastic policy.

    Args:
        mdp: The underlying MDP.
        pol: The stochastic policy.

    Returns:
        Policy, i.e. mapping from state to action.
    """
    pi = {}
    for state in mdp.get_states():
        if mdp.is_terminal(state):
            continue
        pi[state] = pol.get_action(
            state=torch.as_tensor(state, dtype=torch.float32), deterministic=True
        )
    return pi
