from graph_search.a_star import ExampleGraph


def main():
    graph = ExampleGraph()
    graph.draw()

    HH = graph.nodes[0]
    M = graph.nodes[6]

    result = graph.a_star(M, HH)
    print([x.node.name for x in result])

    graph.draw_result(result)


if __name__ == "__main__":
    main()
