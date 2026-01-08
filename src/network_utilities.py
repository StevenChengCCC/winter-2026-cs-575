import networkx as nx  # type: ignore
import numpy as np
from numpy.typing import NDArray
from collections import Counter
from typing import Hashable


####################
## Error Handling ##
####################
class IllegalGraphRepresentation(Exception):
    """Raised when a graph representation is invalid (e.g., empty)."""

    def __init__(self, message: str = "Graph representation had no vertices") -> None:
        super().__init__(message)


######################
## Helper functions ##
######################

def get_degree_count_dictionary(G: nx.Graph) -> dict[int, int]:
    """Return a degree -> count mapping for the graph."""
    degree_list: list[int] = [y for (_, y) in G.degree]
    degree_count: dict[int, int] = dict(Counter(degree_list))
    return degree_count


####################
## Graph Creation ##
####################

def vertex_edge_sets_to_graph(V: set[Hashable], E: tuple[Hashable, Hashable]) -> nx.Graph:
    if len(V) == 0:
        raise IllegalGraphRepresentation("Vertex set cannot be empty")

    # Validate that all edges reference vertices in the vertex set
    for edge in E:
        if not isinstance(edge, tuple):
                raise IllegalGraphRepresentation(f"Edge {edge} is not a tuple")
        if len(edge) != 2:
            raise IllegalGraphRepresentation(
                f"Edge {edge} must contain exactly 2 vertices"
            )
        u, v = edge
        
        if u not in V or v not in V:
            raise IllegalGraphRepresentation(
                f"Edge {edge} contains vertices not in vertex set"
            )

    G = nx.Graph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    return G


def vertex_edge_sets_to_digraph(V: set[int], E: set[tuple[int]]) -> nx.DiGraph:
    if len(V) == 0:
        raise IllegalGraphRepresentation("Vertex set cannot be empty")

    # Validate that all edges reference vertices in the vertex set
    for edge in E:
        u, v = edge
        if u not in V or v not in V:
            raise IllegalGraphRepresentation(
                f"Edge {edge} contains vertices not in vertex set"
            )

    G = nx.DiGraph()
    G.add_nodes_from(V)
    G.add_edges_from(E)
    return G


def adjacency_list_to_graph(adjacency_list: dict[int, set[int]]) -> nx.Graph:
    """Create an undirected graph from an adjacency list."""
    if len(adjacency_list.keys()) == 0:
        raise IllegalGraphRepresentation("Adjacency list had no vertices")

    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            if vertex1 not in adjacency_list[vertex2]:
                raise IllegalGraphRepresentation(
                    "Adjacency list for undirected graph does not have all required edges"
                )

    G = nx.Graph()
    G.add_nodes_from(sorted([vertex for vertex in adjacency_list.keys()]))
    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            G.add_edge(vertex1, vertex2)
    return G


def adjacency_list_to_digraph(adjacency_list: dict[int, set[int]]) -> nx.DiGraph:
    """Create a directed graph from an adjacency list."""
    if len(adjacency_list.keys()) == 0:
        raise IllegalGraphRepresentation("Adjacency list had no vertices")

    G = nx.DiGraph()
    for vertex1 in adjacency_list.keys():
        for vertex2 in adjacency_list[vertex1]:
            G.add_edge(vertex1, vertex2)
    return G


def adjacency_matrix_to_graph(adjacency_matrix: NDArray) -> nx.Graph:
    if len(adjacency_matrix) == 0:
        raise IllegalGraphRepresentation("Adjacency matrix had no vertices")

    if not np.array_equal(adjacency_matrix, adjacency_matrix.T):
        raise IllegalGraphRepresentation("Adjacency matrix is not symmetric")

    return nx.from_numpy_array(adjacency_matrix)


def adjacency_matrix_to_digraph(adjacency_matrix: NDArray) -> nx.Graph:
    if len(adjacency_matrix) == 0:
        raise IllegalGraphRepresentation("Adjacency matrix had no vertices")

    return nx.from_numpy_array(adjacency_matrix, create_using=nx.DiGraph)
