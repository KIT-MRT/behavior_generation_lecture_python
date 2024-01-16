import matplotlib

from behavior_generation_lecture_python.mdp.mdp import (
    GRID_MDP_DICT,
    GridMDP,
    derive_policy,
)
from behavior_generation_lecture_python.utils.grid_plotting import (
    make_plot_grid_step_function,
    make_plot_policy_step_function,
)

TRUE_UTILITY_GRID_MDP = {
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


def test_make_plot_grid_step_function():
    matplotlib.use("Agg")

    plot_grid_step = make_plot_grid_step_function(
        columns=4, rows=3, U_over_time=[TRUE_UTILITY_GRID_MDP]
    )
    plot_grid_step(0)


def test_make_plot_policy_step_function():
    matplotlib.use("Agg")

    policy_array = [
        derive_policy(GridMDP(**GRID_MDP_DICT), utility)
        for utility in [TRUE_UTILITY_GRID_MDP]
    ]
    plot_policy_step = make_plot_policy_step_function(
        columns=4, rows=3, policy_over_time=policy_array
    )
    plot_policy_step(0)
