from network_utilities import adjacency_list_to_graph
import networkx as nx

def test_homework_problem_10() -> None:
    # What I expect
    desired_number_nodes: int = 6
    desired_minimum_density: float = 0.64
    desired_maximum_density: float = 0.68

    # when
    ## FIX THIS ADJACENCY LIST
    adjacency_list: dict[int, set[int]] = {1: {2},
                                           2: {3},
                                           3: {4},
                                           4: {5},
                                           5: {1},
                                           6: {5},
                                           7: {6}}
    G = adjacency_list_to_graph(adjacency_list)

    # then
    assert nx.is_connected(G)
    assert len(G.nodes()) >= desired_number_nodes
    assert nx.density(G) >= desired_minimum_density
    assert nx.density(G) <= desired_maximum_density