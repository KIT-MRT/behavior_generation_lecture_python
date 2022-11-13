import numpy as np

from behavior_generation_lecture_python.graph_search.a_star import Node, Graph


def main():

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

    success = graph.a_star(start="M", end="HH")
    if success:
        graph.draw_result()
    else:
        print(
            "The a start algorithm was not successfull. Maybe check your graph configs."
        )


if __name__ == "__main__":
    main()
