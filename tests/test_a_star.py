import math

import behavior_generation_lecture_python.graph_search.a_star as a_star


class TestNodeClass:
    node1 = a_star.GraphNode("Test Name 1", 0, 0)
    node2 = a_star.GraphNode("Test Name 2", math.sqrt(2), math.sqrt(2))
    node3 = a_star.GraphNode("Test Name 3", 1, 1)

    def test_distance_to_function(self):
        assert self.node1.distance_to(self.node2) == 2

    def test_conntected_to_function(self):
        self.node1.add_connected_to(self.node2)
        self.node1.add_connected_to(self.node3)
        self.node2.add_connected_to(self.node3)
        assert len(self.node1.connected_to) == 2
        assert len(self.node2.connected_to) == 1


def test_example_graph():
    graph = a_star.ExampleGraph()

    HH = graph.nodes[0]
    M = graph.nodes[6]

    result = graph.a_star(M, HH)

    assert [x.node.name for x in result] == ["M", "L", "H", "HH"]
