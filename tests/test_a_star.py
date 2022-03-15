import matplotlib
import numpy as np

from graph_search.a_star import Node, Graph


def test_example_graph():
    matplotlib.use("Agg")

    nodes_list = [
        ["HH", 170, 620, ["H", "B"]],
        ["H", 150, 520, ["B", "L", "F", "HH"]],
        ["B", 330, 540, ["HH", "H", "L"]],
        ["L", 290, 420, ["B", "H", "S", "M"]],
        ["F", 60, 270, ["H", "S"]],
        ["S", 80, 120, ["F", "L", "M"]],
        ["M", 220, 20, ["S", "L"]],
    ]
    nodes_dict = {}
    for entry in nodes_list:
        nodes_dict[entry[0]] = Node(
            name=entry[0],
            position=np.array([entry[1], entry[2]]),
            connected_to=entry[3],
        )

    graph = Graph(nodes_dict=nodes_dict)
    graph.draw_graph()

    graph.a_star(start="M", end="HH")

    graph.draw_result()

    assert [
        (node.name, int(node.cost_to_come), int(node.total_cost()))
        for node in graph.nodes_dict.values()
    ] == [
        ("HH", 680, 680),
        ("H", 578, 680),
        ("B", 532, 711),
        ("L", 406, 639),
        ("F", 323, 690),
        ("S", 172, 680),
        ("M", 0, 602),
    ]
