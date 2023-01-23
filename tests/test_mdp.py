import pytest

from behavior_generation_lecture_python.mdp.mdp import (
    GRID_MDP_DICT,
    MDP,
    SIMPLE_MDP_DICT,
    GridMDP,
    derive_policy,
    expected_utility_of_action,
    value_iteration,
    best_action_from_q_table,
    random_action,
    greedy_estimate_for_state,
    q_learning,
)


def test_init_mdp():
    mdp = MDP(**SIMPLE_MDP_DICT)


def test_init_grid_mdp():
    grid = [[0.1, 0.1, 0.1], [0.2, None, 0.2]]
    grid_mdp = GridMDP(
        grid=grid,
        initial_state=(0, 0),
        terminal_states={(2, 1)},
        transition_probabilities_per_action={(0, 1): [(0.5, (0, 1)), (0.5, (1, 1))]},
    )


def test_expected_utility():
    mdp = MDP(**SIMPLE_MDP_DICT)
    assert 0.8 * 1 + 0.2 * 0.01 == expected_utility_of_action(
        mdp=mdp, state=1, action="A", utility_of_states={1: 0.01, 2: 1}
    )


def test_derive_policy():
    mdp = MDP(**SIMPLE_MDP_DICT)
    expected_policy = {1: "A", 2: None}
    assert expected_policy == derive_policy(mdp=mdp, utility_of_states={1: 0.01, 2: 1})


def test_value_iteration():
    grid_mdp = GridMDP(**GRID_MDP_DICT)
    epsilon = 0.001
    true_utility = {
        (0, 0): 0.705,
        (0, 1): 0.762,
        (0, 2): 0.812,
        (1, 0): 0.655,
        (1, 2): 0.868,
        (2, 0): 0.611,
        (2, 1): 0.660,
        (2, 2): 0.918,
        (3, 0): 0.388,
        (3, 1): -1.0,
        (3, 2): 1.0,
    }

    computed_utility = value_iteration(mdp=grid_mdp, epsilon=epsilon, max_iterations=30)
    for state in true_utility.keys():
        assert abs(true_utility[state] - computed_utility[state]) < epsilon


def test_value_iteration_history():
    grid_mdp = GridMDP(**GRID_MDP_DICT)
    epsilon = 0.001
    true_utility_0 = {
        (0, 0): -0.04,
        (0, 1): -0.04,
        (0, 2): -0.04,
        (1, 0): -0.04,
        (1, 2): -0.04,
        (2, 0): -0.04,
        (2, 1): -0.04,
        (2, 2): -0.04,
        (3, 0): -0.04,
        (3, 1): -1.0,
        (3, 2): 1.0,
    }
    true_utility_1 = {
        (0, 0): -0.08,
        (0, 1): -0.08,
        (0, 2): -0.08,
        (1, 0): -0.08,
        (1, 2): -0.08,
        (2, 0): -0.08,
        (2, 1): -0.08,
        (2, 2): 0.752,
        (3, 0): -0.08,
        (3, 1): -1.0,
        (3, 2): 1.0,
    }
    computed_utility_history = value_iteration(
        mdp=grid_mdp, epsilon=epsilon, max_iterations=30, return_history=True
    )
    for state in true_utility_0.keys():
        assert abs(true_utility_0[state] - computed_utility_history[0][state]) < epsilon

    for state in true_utility_1.keys():
        assert abs(true_utility_1[state] - computed_utility_history[1][state]) < epsilon


def test_best_action_from_q_table():
    q_table = {("A", 1): 0.5, ("A", 2): 0.6, ("B", 1): 0.7, ("B", 2): 0.8}
    avail_actions = {1, 2}
    assert (
        best_action_from_q_table(
            state="A", available_actions=avail_actions, q_table=q_table
        )
        == 2
    )


def test_random_action():
    avail_actions = {1, 2}
    for _ in range(10):
        assert random_action(available_actions=avail_actions) in avail_actions


def test_greedy_estimate_for_state():
    q_table = {("A", 1): 0.5, ("A", 2): 0.6, ("B", 1): 0.7, ("B", 2): 0.8}
    assert greedy_estimate_for_state(q_table=q_table, state="A") == 0.6
    assert greedy_estimate_for_state(q_table=q_table, state="B") == 0.8


@pytest.mark.parametrize("return_history", (True, False))
def test_q_learning(return_history):
    assert q_learning(
        mdp=GridMDP(**GRID_MDP_DICT),
        alpha=0.1,
        epsilon=0.1,
        iterations=10000,
        return_history=return_history,
    )
