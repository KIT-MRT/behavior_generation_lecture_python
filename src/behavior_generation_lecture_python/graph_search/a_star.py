from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, List, Set


class Node:
    def __init__(
        self, name: str, position: np.ndarray, connected_to: List[str]
    ) -> None:
        """
        Node in a graph for A* computation.

        :param name: Name of the node.
        :param position: Position of the node (x,y).
        :param connected_to: List of the names of nodes, that this node is connected to.
        """
        self.name = name
        self.position = position
        self.connected_to = connected_to
        self.predecessor = None
        self.cost_to_come = None
        self.heuristic_cost_to_go = None

    def compute_heuristic_cost_to_go(self, goal_node: Node) -> None:
        """
        Computes the heuristic cost to go to the goal node based on the distance and assigns it to the node object.

        :param goal_node: The goal node.
        :return:
        """
        self.heuristic_cost_to_go = np.linalg.norm(goal_node.position - self.position)

    def total_cost(self) -> float:
        """
        Computes the expected total cost to reach the goal node as sum of cost to come and heuristic cost to go.

        :return: The expected total cost to reach the goal node.
        """
        return self.cost_to_come + self.heuristic_cost_to_go


def extract_min(node_set: Set[str], node_dict: Dict[str, Node]) -> str:
    """
    Extract the node with minimal total cost from a set.

    :param node_set: The set of node names to be considered.
    :param node_dict: The node dict, containing the node information.
    :return: The name of the node with minimal total cost.
    """
    min_node = min(node_set, key=lambda x: node_dict[x].total_cost())
    node_set.remove(min_node)
    return min_node


class Graph:
    def __init__(self, nodes_dict: Dict[str, Node]) -> None:
        """
        A graph for A* computation.

        :param nodes_dict: The dictionary containing the nodes of the graph.
        """
        self.nodes_dict = nodes_dict

        self._end_node = None

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

    def draw_graph(self) -> None:
        """
        Draw all nodes and their connections in the graph.

        :return:
        """
        self.ax.set_xlim([0, 700])
        self.ax.set_ylim([0, 700])

        for node in self.nodes_dict.values():
            self.ax.plot(
                node.position[0], node.position[1], marker="o", markersize=6, color="k"
            )
            self.ax.annotate(node.name, (node.position[0] + 10, node.position[1] + 10))
            for connected_to in node.connected_to:
                connected_node = self.nodes_dict[connected_to]
                self.ax.plot(
                    [node.position[0], connected_node.position[0]],
                    [node.position[1], connected_node.position[1]],
                    linewidth=1,
                    color="k",
                )

    def draw_result(self) -> None:
        """
        Draw the solution to the shortest path problem.

        :return:
        """
        assert (
            self._end_node
        ), "End node not defined, run a_star() before drawing the result."
        current_node = self._end_node
        while self.nodes_dict[current_node].predecessor:
            curr_node = self.nodes_dict[current_node]
            predecessor = curr_node.predecessor
            pred_node = self.nodes_dict[predecessor]

            self.ax.plot(
                [curr_node.position[0], pred_node.position[0]],
                [curr_node.position[1], pred_node.position[1]],
                linewidth=2,
                color="b",
            )

            distance = np.linalg.norm(curr_node.position - pred_node.position)
            x_mid = (curr_node.position[0] + pred_node.position[0]) / 2.0
            y_mid = (curr_node.position[1] + pred_node.position[1]) / 2.0
            self.ax.annotate(f"{distance:.2f}", (x_mid + 10, y_mid + 10))
            current_node = predecessor

        plt.show()

    def a_star(self, start: str, end: str) -> bool:
        """
        Compute the shortest path through the graph with the A* algorithm.

        :param start: Name of the start node.
        :param end: Name of the end node.
        :return: True if shortest path found, False otherwise.
        """
        try:
            self.nodes_dict[start]
            self.nodes_dict[end]
        except KeyError as error:
            print(
                f"Could not find node {error} in the graph, Make sure that the start and end nodes are in the graph."
            )
            return False
        self._end_node = end
        open_set = set()
        closed_set = set()
        for node in self.nodes_dict.values():
            node.compute_heuristic_cost_to_go(self.nodes_dict[end])

        open_set.add(start)
        self.nodes_dict[start].cost_to_come = 0

        while open_set:
            current_node = extract_min(node_set=open_set, node_dict=self.nodes_dict)

            if current_node == end:
                return True

            closed_set.add(current_node)

            for successor_node in self.nodes_dict[current_node].connected_to:
                if successor_node in closed_set:
                    continue

                tentative_cost_to_come = self.nodes_dict[
                    current_node
                ].cost_to_come + np.linalg.norm(
                    self.nodes_dict[current_node].position
                    - self.nodes_dict[successor_node].position
                )
                if (
                    successor_node in open_set
                    and tentative_cost_to_come
                    >= self.nodes_dict[successor_node].cost_to_come
                ):
                    continue

                self.nodes_dict[successor_node].predecessor = current_node
                self.nodes_dict[successor_node].cost_to_come = tentative_cost_to_come
                open_set.add(successor_node)

        return False
