import networkx as nx # type: ignore
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


def get_principal_eigenvector_undirected(G: nx.Graph) -> Tuple[float, NDArray[np.floating]]:
    """
    Compute the principal eigenvector and eigenvalue for an undirected graph.
    
    The principal eigenvector corresponds to the largest eigenvalue of the 
    adjacency matrix and is used for eigenvector centrality calculations.
    For undirected graphs, the Perron-Frobenius theorem guarantees that the
    principal eigenvalue is real, positive, and has the largest magnitude.
    
    Parameters
    ----------
    G : nx.Graph
        An undirected NetworkX graph. Must be connected.
        
    Returns
    -------
    principal_eigenvalue : float
        The largest eigenvalue of the adjacency matrix
    principal_eigenvector : NDArray
        The eigenvector corresponding to the largest eigenvalue
        
    Raises
    ------
    TypeError
        If G is not an undirected graph (nx.Graph)
    ValueError
        If the graph is not connected
        
    Examples
    --------
    >>> G = nx.karate_club_graph()
    >>> eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
    """
    # Check that the graph is undirected (not a DiGraph)
    if isinstance(G, nx.DiGraph):
        raise TypeError("Graph must be undirected (nx.Graph), not directed (nx.DiGraph)")
    
    # Check that the graph is connected
    if not nx.is_connected(G):
        raise ValueError("Undirected graph must be connected")
    
    # Get the adjacency matrix with nodes in sorted order
    A: NDArray[np.floating] = nx.adjacency_matrix(
        G, 
        nodelist=[node for node in sorted(G.nodes)]
    ).toarray()
    
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Find the index of the principal eigenvalue (the largest eigenvalue by magnitude)
    principal_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    
    # Get the principal eigenvalue
    principal_eigenvalue = eigenvalues[principal_eigenvalue_index]
    
    # Get the principal eigenvector
    principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]
    
    return principal_eigenvalue, principal_eigenvector


def get_principal_eigenvector_directed(G: nx.DiGraph) -> Tuple[float, NDArray[np.floating]]:
    """
    Compute the principal eigenvector and eigenvalue for a directed graph.
    
    The principal eigenvector corresponds to the largest eigenvalue of the 
    transpose of the adjacency matrix and is used for eigenvector centrality 
    calculations in directed graphs. For strongly connected directed graphs,
    the Perron-Frobenius theorem guarantees that the principal eigenvalue is
    real, positive, and has the largest magnitude.
    
    Parameters
    ----------
    G : nx.DiGraph
        A directed NetworkX graph. Must be strongly connected.
        
    Returns
    -------
    principal_eigenvalue : float
        The largest eigenvalue of the transpose adjacency matrix
    principal_eigenvector : NDArray
        The eigenvector corresponding to the largest eigenvalue
        
    Raises
    ------
    TypeError
        If G is not a directed graph (nx.DiGraph)
    ValueError
        If the graph is not strongly connected
        
    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    >>> eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
    """
    # Check that the graph is directed
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph must be a directed graph (nx.DiGraph)")
    
    # Check that the graph is strongly connected
    if not nx.is_strongly_connected(G):
        raise ValueError("Directed graph must be strongly connected")
    
    # Get the adjacency matrix with nodes in sorted order
    A: NDArray[np.floating] = nx.adjacency_matrix(
        G, 
        nodelist=[node for node in sorted(G.nodes)]
    ).toarray()
    
    # Compute the eigenvalues and eigenvectors of the transpose
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    
    # Find the index of the principal eigenvalue (the largest eigenvalue by magnitude)
    principal_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    
    # Get the principal eigenvalue
    principal_eigenvalue = eigenvalues[principal_eigenvalue_index]
    
    # Get the principal eigenvector
    principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]
    
    return principal_eigenvalue, principal_eigenvector


def get_principal_eigenvector_directed_unchecked(G: nx.DiGraph) -> Tuple[float, NDArray[np.floating]]:
    """
    Compute the principal eigenvector and eigenvalue for a directed graph WITHOUT checking connectivity.
    
    This function computes the principal eigenvector of the transpose of the adjacency
    matrix without verifying that the graph is strongly connected. This is useful for
    educational purposes to demonstrate what happens with graphs that have dead-ends
    or spider traps.
    
    WARNING: For graphs that are not strongly connected, the Perron-Frobenius theorem
    does not guarantee that the principal eigenvalue will be real, positive, and have
    the largest magnitude. Use with caution.
    
    Parameters
    ----------
    G : nx.DiGraph
        A directed NetworkX graph. May or may not be strongly connected.
        
    Returns
    -------
    principal_eigenvalue : float
        The largest eigenvalue (by magnitude) of the transpose adjacency matrix
    principal_eigenvector : NDArray
        The eigenvector corresponding to the largest eigenvalue
        
    Raises
    ------
    TypeError
        If G is not a directed graph (nx.DiGraph)
        
    Examples
    --------
    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (2, 3)])  # Not strongly connected
    >>> eigenvalue, eigenvector = get_principal_eigenvector_directed_unchecked(G)
    """
    # Check that the graph is directed
    if not isinstance(G, nx.DiGraph):
        raise TypeError("Graph must be a directed graph (nx.DiGraph)")
    
    # Get the adjacency matrix with nodes in sorted order
    A: NDArray[np.floating] = nx.adjacency_matrix(
        G, 
        nodelist=[node for node in sorted(G.nodes)]
    ).toarray()
    
    # Compute the eigenvalues and eigenvectors of the transpose
    eigenvalues, eigenvectors = np.linalg.eig(A.T)
    
    # Find the index of the principal eigenvalue (the largest eigenvalue by magnitude)
    principal_eigenvalue_index = np.argmax(np.abs(eigenvalues))
    
    # Get the principal eigenvalue
    principal_eigenvalue = eigenvalues[principal_eigenvalue_index]
    
    # Get the principal eigenvector
    principal_eigenvector = eigenvectors[:, principal_eigenvalue_index]
    
    return principal_eigenvalue, principal_eigenvector
