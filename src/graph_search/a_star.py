import matplotlib.pyplot as plt
import numpy as np


class Node:
    def __init__(self, name, position, connected_to):
        self.name = name
        self.position = position
        self.connected_to = connected_to
        self.predecessor = None
        self.cost_to_come = None
        self.heuristic_cost_to_go = None

    def compute_heuristic_cost_to_go(self, goal_node):
        self.heuristic_cost_to_go = np.linalg.norm(goal_node.position - self.position)

    def total_cost(self):
        return self.cost_to_come + self.heuristic_cost_to_go


def extract_min(node_set, node_dict):
    min_node = min(node_set, key=lambda x: node_dict[x].total_cost())
    node_set.remove(min_node)
    return min_node


class Graph:
    def __init__(self, nodes_dict) -> None:
        self.nodes_dict = nodes_dict

        self.end_node = None

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

    def draw_graph(self):
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

    def draw_result(self):
        current_node = self.end_node
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

    def a_star(self, start, end):
        """
        returns False if unsuccessful
        """
        self.end_node = end
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
