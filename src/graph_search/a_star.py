import matplotlib.pyplot as plt
import math


class GraphNode:
    def __init__(self, name, x, y):
        self.name = name
        self.x = x
        self.y = y
        self.connected_to = []

    def add_connected_to(self, connected_to):
        self.connected_to.append(connected_to)

    def distance_to(self, node):
        return math.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)


class AStarNode:
    def __init__(self, node, end_node):
        self.C = 0
        self.G = node.distance_to(end_node)
        self.J = 0
        self.node = node
        self.predecessor = None


class ExampleGraph:
    def __init__(self):
        HH = GraphNode("HH", 170, 620)
        H = GraphNode("H", 150, 520)
        B = GraphNode("B", 330, 540)
        L = GraphNode("L", 290, 420)
        F = GraphNode("F", 60, 270)
        S = GraphNode("S", 80, 120)
        M = GraphNode("M", 220, 20)

        self.nodes = [HH, H, B, L, F, S, M]

        HH.add_connected_to(H)
        H.add_connected_to(HH)
        HH.add_connected_to(B)
        B.add_connected_to(HH)

        H.add_connected_to(B)
        B.add_connected_to(H)
        H.add_connected_to(L)
        L.add_connected_to(H)
        H.add_connected_to(F)
        F.add_connected_to(H)

        B.add_connected_to(L)
        L.add_connected_to(B)

        L.add_connected_to(S)
        S.add_connected_to(L)
        L.add_connected_to(M)
        M.add_connected_to(L)

        F.add_connected_to(S)
        S.add_connected_to(F)

        S.add_connected_to(M)
        M.add_connected_to(S)

        fig, ax = plt.subplots()
        self.fig = fig
        self.ax = ax

    def draw(self):
        self.ax.set_xlim([0, 700])
        self.ax.set_ylim([0, 700])

        for node in self.nodes:
            self.ax.plot(node.x, node.y, marker="o", markersize=6, color="k")
            self.ax.annotate(node.name, (node.x + 10, node.y + 10))
            for connected_to in node.connected_to:
                self.ax.plot(
                    [node.x, connected_to.x],
                    [node.y, connected_to.y],
                    linewidth=1,
                    color="k",
                )

    def a_star(self, start, end):
        open_set = []
        closed_set = []

        x_0 = AStarNode(start, end)

        x_0.J = x_0.G

        open_set.append(x_0)

        def extract_min(open_set):
            if open_set:
                result = open_set[0]
                min_J = result.J
                for node in open_set:
                    if node.J < min_J:
                        result = node
                        min_J = result.J

                open_set.remove(result)
                return result

            return None

        def retrace_path(node):
            path = [node]

            while node.predecessor is not None:
                node = node.predecessor
                path.append(node)

            path.reverse()
            return path

        while open_set:
            x = extract_min(open_set)
            closed_set.append(x)

            if x.node == end:
                return retrace_path(x)
            else:
                for node in x.node.connected_to:
                    x_tilde = AStarNode(node, end)

                    if x_tilde.node not in closed_set:
                        cost = x.C + x_tilde.node.distance_to(x.node)

                        if (
                            not x_tilde.node in [x.node for x in open_set]
                            or cost < x_tilde.G
                        ):
                            x_tilde.predecessor = x
                            x_tilde.C = cost
                            x_tilde.J = x_tilde.C + x_tilde.G

                            if not x_tilde.node in [x.node for x in open_set]:
                                open_set.append(x_tilde)

    def draw_result(self, result):
        for i in range(len(result) - 1):
            node_from = result[i].node
            node_to = result[i + 1].node
            self.ax.plot(
                [node_from.x, node_to.x],
                [node_from.y, node_to.y],
                linewidth=2,
                color="b",
            )

            distance = node_from.distance_to(node_to)
            x_mid = (node_from.x + node_to.x) / 2
            y_mid = (node_from.y + node_to.y) / 2
            self.ax.annotate(f"{distance:.2f}", (x_mid + 10, y_mid + 10))

        plt.show()
