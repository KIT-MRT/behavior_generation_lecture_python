import numpy as np

from behavior_generation_lecture_python.mdp.mdp import (
    GRID_MDP_DICT,
    HIGHWAY_MDP_DICT,
    LC_RIGHT_ACTION,
    MDP,
    STAY_IN_LANE_ACTION,
    GridMDP,
    derive_policy,
    expected_utility_of_action,
    value_iteration,
)


def values_to_tex(value_per_state: dict, node_property: str = "") -> str:
    tex_list = []
    for position, value in sorted(value_per_state.items()):
        if value:
            tex_list.append(
                r"\node"
                + node_property
                + " at "
                + str(tuple(np.array(position) + np.array((0.5, 0.5))))
                + r" {$"
                + "{:.3g}".format(value)
                + r"$};"
            )
    return "\n".join(tex_list)


def rewards_to_tex(mdp: MDP, node_property: str = "") -> str:
    tex_list = []
    for position in sorted(mdp.get_states()):
        if mdp.get_reward(position) is not None:
            tex_list.append(
                r"\node"
                + node_property
                + " at "
                + str(tuple(np.array(position) + np.array((0.5, 0.5))))
                + r" {$"
                + "{:.3g}".format(mdp.get_reward(position))
                + r"$};"
            )
    return "\n".join(tex_list)


def expected_utilities_to_tex(mdp: MDP, utility: dict) -> str:

    exp_util = {}
    for s in sorted(mdp.get_states()):
        for a in sorted(mdp.get_actions(s)):
            if not a:
                continue
            exp_util[
                tuple(np.array(s) + 0.35 * np.array(a))
            ] = expected_utility_of_action(
                action=a, state=s, utility_of_states=utility, mdp=mdp
            )

    return values_to_tex(exp_util, node_property="[scale=0.8]")


def best_policy_to_tex_arrows(mdp: MDP, utility: dict):

    source_dest_list = []
    policy = derive_policy(mdp=mdp, utility_of_states=utility)
    for s in sorted(mdp.get_states()):
        best_action = policy[s]
        if not best_action:
            continue
        source = tuple(np.array(s) - 0.2 * np.array(best_action) + np.array((0.5, 0.5)))
        dest = tuple(np.array(s) + 0.2 * np.array(best_action) + np.array((0.5, 0.5)))
        source_dest_list.append((source, dest))

    tex_list = []
    for source, dest in source_dest_list:
        tex_list.append(
            r"\draw[->,color=blue] " + str(source) + " -- " + str(dest) + r";"
        )
    return "\n".join(tex_list)


GRID_MPD_REWARDS_TEX = r"""\node at (0.5, 0.5) {$-0.04$};
\node at (0.5, 1.5) {$-0.04$};
\node at (0.5, 2.5) {$-0.04$};
\node at (1.5, 0.5) {$-0.04$};
\node at (1.5, 2.5) {$-0.04$};
\node at (2.5, 0.5) {$-0.04$};
\node at (2.5, 1.5) {$-0.04$};
\node at (2.5, 2.5) {$-0.04$};
\node at (3.5, 0.5) {$-0.04$};
\node at (3.5, 1.5) {$-1$};
\node at (3.5, 2.5) {$1$};"""

GRID_MDP_TRUE_UTILITY_TEX = r"""\node at (0.5, 0.5) {$0.705$};
\node at (0.5, 1.5) {$0.762$};
\node at (0.5, 2.5) {$0.812$};
\node at (1.5, 0.5) {$0.655$};
\node at (1.5, 2.5) {$0.868$};
\node at (2.5, 0.5) {$0.611$};
\node at (2.5, 1.5) {$0.66$};
\node at (2.5, 2.5) {$0.918$};
\node at (3.5, 0.5) {$0.388$};
\node at (3.5, 1.5) {$-1$};
\node at (3.5, 2.5) {$1$};"""


def test_latex_value_over_time():

    grid_mdp = GridMDP(**GRID_MDP_DICT)
    assert GRID_MPD_REWARDS_TEX == rewards_to_tex(grid_mdp)

    utility = value_iteration(grid_mdp, epsilon=0.0001, max_iterations=30)
    assert GRID_MDP_TRUE_UTILITY_TEX == values_to_tex(utility)


GRID_MDP_EXPECTED_UTILITIES_PER_ACTION_TEX = r"""\node[scale=0.8] at (0.15000000000000002, 0.5) {$0.711$};
\node[scale=0.8] at (0.15000000000000002, 1.5) {$0.761$};
\node[scale=0.8] at (0.15000000000000002, 2.5) {$0.807$};
\node[scale=0.8] at (0.5, 0.15000000000000002) {$0.7$};
\node[scale=0.8] at (0.5, 0.85) {$0.745$};
\node[scale=0.8] at (0.5, 1.15) {$0.717$};
\node[scale=0.8] at (0.5, 1.85) {$0.802$};
\node[scale=0.8] at (0.5, 2.15) {$0.777$};
\node[scale=0.8] at (0.5, 2.85) {$0.817$};
\node[scale=0.8] at (0.85, 0.5) {$0.671$};
\node[scale=0.8] at (0.85, 1.5) {$0.761$};
\node[scale=0.8] at (0.85, 2.5) {$0.852$};
\node[scale=0.8] at (1.15, 0.5) {$0.695$};
\node[scale=0.8] at (1.15, 2.5) {$0.823$};
\node[scale=0.8] at (1.5, 0.15000000000000002) {$0.656$};
\node[scale=0.8] at (1.5, 0.85) {$0.656$};
\node[scale=0.8] at (1.5, 2.15) {$0.867$};
\node[scale=0.8] at (1.5, 2.85) {$0.867$};
\node[scale=0.8] at (1.85, 0.5) {$0.62$};
\node[scale=0.8] at (1.85, 2.5) {$0.908$};
\node[scale=0.8] at (2.15, 0.5) {$0.651$};
\node[scale=0.8] at (2.15, 1.5) {$0.681$};
\node[scale=0.8] at (2.15, 2.5) {$0.852$};
\node[scale=0.8] at (2.5, 0.15000000000000002) {$0.593$};
\node[scale=0.8] at (2.5, 0.85) {$0.633$};
\node[scale=0.8] at (2.5, 1.15) {$0.455$};
\node[scale=0.8] at (2.5, 1.85) {$0.7$};
\node[scale=0.8] at (2.5, 2.15) {$0.715$};
\node[scale=0.8] at (2.5, 2.85) {$0.921$};
\node[scale=0.8] at (2.85, 0.5) {$0.437$};
\node[scale=0.8] at (2.85, 1.5) {$-0.647$};
\node[scale=0.8] at (2.85, 2.5) {$0.958$};
\node[scale=0.8] at (3.15, 0.5) {$0.428$};
\node[scale=0.8] at (3.5, 0.15000000000000002) {$0.41$};
\node[scale=0.8] at (3.5, 0.85) {$-0.7$};
\node[scale=0.8] at (3.85, 0.5) {$0.249$};"""


def test_latex_expected_utility():

    grid_mdp = GridMDP(**GRID_MDP_DICT)

    utility = value_iteration(grid_mdp, epsilon=0.0001, max_iterations=30)

    exp_util_tex = expected_utilities_to_tex(
        grid_mdp,
        utility,
    )

    assert exp_util_tex == GRID_MDP_EXPECTED_UTILITIES_PER_ACTION_TEX


GRID_MDP_BEST_POLICY_AS_ARROWS_TEX = r"""\draw[->,color=blue] (0.5, 0.3) -- (0.5, 0.7);
\draw[->,color=blue] (0.5, 1.3) -- (0.5, 1.7);
\draw[->,color=blue] (0.3, 2.5) -- (0.7, 2.5);
\draw[->,color=blue] (1.7, 0.5) -- (1.3, 0.5);
\draw[->,color=blue] (1.3, 2.5) -- (1.7, 2.5);
\draw[->,color=blue] (2.7, 0.5) -- (2.3, 0.5);
\draw[->,color=blue] (2.5, 1.3) -- (2.5, 1.7);
\draw[->,color=blue] (2.3, 2.5) -- (2.7, 2.5);
\draw[->,color=blue] (3.7, 0.5) -- (3.3, 0.5);"""


def test_latex_policy_as_arrows():
    grid_mdp = GridMDP(**GRID_MDP_DICT)

    utility = value_iteration(grid_mdp, epsilon=0.0001, max_iterations=30)

    tex_text = best_policy_to_tex_arrows(
        grid_mdp,
        utility,
    )
    assert tex_text == GRID_MDP_BEST_POLICY_AS_ARROWS_TEX


HIGHWAY_MDP_REWARDS_TEX = r"""\node at (0.5, 1.5) {$0$};
\node at (0.5, 2.5) {$0$};
\node at (0.5, 3.5) {$0$};
\node at (1.5, 1.5) {$-3$};
\node at (1.5, 2.5) {$-2$};
\node at (1.5, 3.5) {$-1$};
\node at (2.5, 1.5) {$-3$};
\node at (2.5, 2.5) {$-2$};
\node at (2.5, 3.5) {$-1$};
\node at (3.5, 1.5) {$-3$};
\node at (3.5, 2.5) {$-2$};
\node at (3.5, 3.5) {$-1$};
\node at (4.5, 1.5) {$-3$};
\node at (4.5, 2.5) {$-2$};
\node at (4.5, 3.5) {$-1$};
\node at (5.5, 1.5) {$-3$};
\node at (5.5, 2.5) {$-2$};
\node at (5.5, 3.5) {$-1$};
\node at (6.5, 0.5) {$-2$};
\node at (6.5, 1.5) {$-3$};
\node at (6.5, 2.5) {$-2$};
\node at (6.5, 3.5) {$-1$};
\node at (7.5, 0.5) {$-2$};
\node at (7.5, 1.5) {$-3$};
\node at (7.5, 2.5) {$-2$};
\node at (7.5, 3.5) {$-1$};
\node at (8.5, 0.5) {$-2$};
\node at (8.5, 1.5) {$-3$};
\node at (8.5, 2.5) {$-2$};
\node at (8.5, 3.5) {$-1$};
\node at (9.5, 0.5) {$0$};
\node at (9.5, 1.5) {$-50$};
\node at (9.5, 2.5) {$-50$};
\node at (9.5, 3.5) {$-50$};"""

HIGHWAY_MDP_UTILITY_TEX = r"""\node at (0.5, 1.5) {$-18.6$};
\node at (0.5, 2.5) {$-16.6$};
\node at (0.5, 3.5) {$-15.8$};
\node at (1.5, 1.5) {$-19.8$};
\node at (1.5, 2.5) {$-17.4$};
\node at (1.5, 3.5) {$-15.8$};
\node at (2.5, 1.5) {$-17.7$};
\node at (2.5, 2.5) {$-16$};
\node at (2.5, 3.5) {$-14.8$};
\node at (3.5, 1.5) {$-15.3$};
\node at (3.5, 2.5) {$-14.1$};
\node at (3.5, 3.5) {$-13.8$};
\node at (4.5, 1.5) {$-12.5$};
\node at (4.5, 2.5) {$-12.1$};
\node at (4.5, 3.5) {$-15$};
\node at (5.5, 1.5) {$-9.52$};
\node at (5.5, 2.5) {$-11.8$};
\node at (5.5, 3.5) {$-20.7$};
\node at (6.5, 0.5) {$-6$};
\node at (6.5, 1.5) {$-8.09$};
\node at (6.5, 2.5) {$-14.9$};
\node at (6.5, 3.5) {$-34$};
\node at (7.5, 0.5) {$-4$};
\node at (7.5, 1.5) {$-8.38$};
\node at (7.5, 2.5) {$-26.6$};
\node at (7.5, 3.5) {$-52$};
\node at (8.5, 0.5) {$-2$};
\node at (8.5, 1.5) {$-15.5$};
\node at (8.5, 2.5) {$-52$};
\node at (8.5, 3.5) {$-51$};
\node at (9.5, 1.5) {$-50$};
\node at (9.5, 2.5) {$-50$};
\node at (9.5, 3.5) {$-50$};"""

HIGHWAY_MDP_OPTIMAL_POLICY_ARROWS_TEX = r"""\draw[->,color=blue] (0.3, 1.3) -- (0.7, 1.7);
\draw[->,color=blue] (0.3, 2.3) -- (0.7, 2.7);
\draw[->,color=blue] (0.3, 3.5) -- (0.7, 3.5);
\draw[->,color=blue] (1.3, 1.3) -- (1.7, 1.7);
\draw[->,color=blue] (1.3, 2.3) -- (1.7, 2.7);
\draw[->,color=blue] (1.3, 3.5) -- (1.7, 3.5);
\draw[->,color=blue] (2.3, 1.3) -- (2.7, 1.7);
\draw[->,color=blue] (2.3, 2.3) -- (2.7, 2.7);
\draw[->,color=blue] (2.3, 3.5) -- (2.7, 3.5);
\draw[->,color=blue] (3.3, 1.3) -- (3.7, 1.7);
\draw[->,color=blue] (3.3, 2.5) -- (3.7, 2.5);
\draw[->,color=blue] (3.3, 3.7) -- (3.7, 3.3);
\draw[->,color=blue] (4.3, 1.5) -- (4.7, 1.5);
\draw[->,color=blue] (4.3, 2.7) -- (4.7, 2.3);
\draw[->,color=blue] (4.3, 3.7) -- (4.7, 3.3);
\draw[->,color=blue] (5.3, 1.7) -- (5.7, 1.3);
\draw[->,color=blue] (5.3, 2.7) -- (5.7, 2.3);
\draw[->,color=blue] (5.3, 3.7) -- (5.7, 3.3);
\draw[->,color=blue] (6.3, 0.5) -- (6.7, 0.5);
\draw[->,color=blue] (6.3, 1.7) -- (6.7, 1.3);
\draw[->,color=blue] (6.3, 2.7) -- (6.7, 2.3);
\draw[->,color=blue] (6.3, 3.7) -- (6.7, 3.3);
\draw[->,color=blue] (7.3, 0.5) -- (7.7, 0.5);
\draw[->,color=blue] (7.3, 1.7) -- (7.7, 1.3);
\draw[->,color=blue] (7.3, 2.7) -- (7.7, 2.3);
\draw[->,color=blue] (7.3, 3.5) -- (7.7, 3.5);
\draw[->,color=blue] (8.3, 0.5) -- (8.7, 0.5);
\draw[->,color=blue] (8.3, 1.7) -- (8.7, 1.3);
\draw[->,color=blue] (8.3, 2.5) -- (8.7, 2.5);
\draw[->,color=blue] (8.3, 3.5) -- (8.7, 3.5);"""

HIGHWAY_MDP_LC_R_0DOT4_UTILITY_TEX = r"""\node at (0.5, 1.5) {$-28.6$};
\node at (0.5, 2.5) {$-28.6$};
\node at (0.5, 3.5) {$-31$};
\node at (1.5, 1.5) {$-28.7$};
\node at (1.5, 2.5) {$-28.6$};
\node at (1.5, 3.5) {$-32.6$};
\node at (2.5, 1.5) {$-25.7$};
\node at (2.5, 2.5) {$-27.2$};
\node at (2.5, 3.5) {$-34.5$};
\node at (3.5, 1.5) {$-22.7$};
\node at (3.5, 2.5) {$-27$};
\node at (3.5, 3.5) {$-37.9$};
\node at (4.5, 1.5) {$-19.7$};
\node at (4.5, 2.5) {$-28.5$};
\node at (4.5, 3.5) {$-42.5$};
\node at (5.5, 1.5) {$-16.7$};
\node at (5.5, 2.5) {$-33.1$};
\node at (5.5, 3.5) {$-47.2$};
\node at (6.5, 0.5) {$-6$};
\node at (6.5, 1.5) {$-18.8$};
\node at (6.5, 2.5) {$-39.3$};
\node at (6.5, 3.5) {$-50.8$};
\node at (7.5, 0.5) {$-4$};
\node at (7.5, 1.5) {$-23.6$};
\node at (7.5, 2.5) {$-46.4$};
\node at (7.5, 3.5) {$-52$};
\node at (8.5, 0.5) {$-2$};
\node at (8.5, 1.5) {$-33$};
\node at (8.5, 2.5) {$-52$};
\node at (8.5, 3.5) {$-51$};
\node at (9.5, 1.5) {$-50$};
\node at (9.5, 2.5) {$-50$};
\node at (9.5, 3.5) {$-50$};"""

HIGHWAY_MDP_LC_R_0DOT4_OPTIMAL_POLICY_ARROWS_TEX = r"""\draw[->,color=blue] (0.3, 1.3) -- (0.7, 1.7);
\draw[->,color=blue] (0.3, 2.5) -- (0.7, 2.5);
\draw[->,color=blue] (0.3, 3.7) -- (0.7, 3.3);
\draw[->,color=blue] (1.3, 1.5) -- (1.7, 1.5);
\draw[->,color=blue] (1.3, 2.7) -- (1.7, 2.3);
\draw[->,color=blue] (1.3, 3.7) -- (1.7, 3.3);
\draw[->,color=blue] (2.3, 1.5) -- (2.7, 1.5);
\draw[->,color=blue] (2.3, 2.7) -- (2.7, 2.3);
\draw[->,color=blue] (2.3, 3.7) -- (2.7, 3.3);
\draw[->,color=blue] (3.3, 1.5) -- (3.7, 1.5);
\draw[->,color=blue] (3.3, 2.7) -- (3.7, 2.3);
\draw[->,color=blue] (3.3, 3.7) -- (3.7, 3.3);
\draw[->,color=blue] (4.3, 1.5) -- (4.7, 1.5);
\draw[->,color=blue] (4.3, 2.7) -- (4.7, 2.3);
\draw[->,color=blue] (4.3, 3.7) -- (4.7, 3.3);
\draw[->,color=blue] (5.3, 1.7) -- (5.7, 1.3);
\draw[->,color=blue] (5.3, 2.7) -- (5.7, 2.3);
\draw[->,color=blue] (5.3, 3.7) -- (5.7, 3.3);
\draw[->,color=blue] (6.3, 0.5) -- (6.7, 0.5);
\draw[->,color=blue] (6.3, 1.7) -- (6.7, 1.3);
\draw[->,color=blue] (6.3, 2.7) -- (6.7, 2.3);
\draw[->,color=blue] (6.3, 3.7) -- (6.7, 3.3);
\draw[->,color=blue] (7.3, 0.5) -- (7.7, 0.5);
\draw[->,color=blue] (7.3, 1.7) -- (7.7, 1.3);
\draw[->,color=blue] (7.3, 2.7) -- (7.7, 2.3);
\draw[->,color=blue] (7.3, 3.5) -- (7.7, 3.5);
\draw[->,color=blue] (8.3, 0.5) -- (8.7, 0.5);
\draw[->,color=blue] (8.3, 1.7) -- (8.7, 1.3);
\draw[->,color=blue] (8.3, 2.5) -- (8.7, 2.5);
\draw[->,color=blue] (8.3, 3.5) -- (8.7, 3.5);"""


def test_latex_value_over_time_highway():

    highway_mdp = GridMDP(**HIGHWAY_MDP_DICT)
    rewards_tex = rewards_to_tex(highway_mdp)
    print(rewards_tex)
    assert rewards_tex == HIGHWAY_MDP_REWARDS_TEX

    utility_history = value_iteration(
        highway_mdp, epsilon=0.001, max_iterations=30, return_history=True
    )
    assert HIGHWAY_MDP_UTILITY_TEX == values_to_tex(utility_history[-1])

    tex_text_arrows = best_policy_to_tex_arrows(
        highway_mdp,
        utility_history[-1],
    )

    assert tex_text_arrows == HIGHWAY_MDP_OPTIMAL_POLICY_ARROWS_TEX

    highway_mdp_lc_right_0dot4_cf = HIGHWAY_MDP_DICT.copy()
    highway_mdp_lc_right_0dot4_cf["transition_probabilities_per_action"][
        LC_RIGHT_ACTION
    ] = [
        (0.4, LC_RIGHT_ACTION),
        (0.6, STAY_IN_LANE_ACTION),
    ]

    highway_mdp_lc_right_0dot4 = GridMDP(**highway_mdp_lc_right_0dot4_cf)
    utility_history_lc_right_0dot4 = value_iteration(
        highway_mdp_lc_right_0dot4,
        epsilon=0.001,
        max_iterations=30,
        return_history=True,
    )

    assert (
        values_to_tex(utility_history_lc_right_0dot4[-1])
        == HIGHWAY_MDP_LC_R_0DOT4_UTILITY_TEX
    )

    tex_text_lc_right0dot4 = best_policy_to_tex_arrows(
        highway_mdp_lc_right_0dot4,
        utility_history_lc_right_0dot4[-1],
    )

    assert tex_text_lc_right0dot4 == HIGHWAY_MDP_LC_R_0DOT4_OPTIMAL_POLICY_ARROWS_TEX
