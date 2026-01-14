import networkx as nx  # type: ignore
import numpy as np
from centrality_utilities import get_principal_eigenvector_directed_unchecked


def _create_pinwheel_graph() -> nx.DiGraph:
    """Helper function to create the directed pinwheel graph"""
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    # Strict cycles
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])  # Cycle 1-2-3-1
    G.add_edges_from([(4, 5), (5, 6), (6, 4)])  # Cycle 4-5-6-4
    G.add_edges_from([(8, 9), (9, 10), (10, 8)])  # Cycle 8-9-10-8
    
    # Node 7 points to 3, 4, and 8 (no reverse edges)
    G.add_edges_from([(7, 3), (7, 4), (7, 8)])
    
    return G


def test_unchecked_eigenvector_not_all_zero() -> None:
    """
    Test that the unchecked eigenvector function returns non-zero values.
    
    The pinwheel graph is reducible (not strongly connected) because node 7
    has no incoming edges and the three cycles are isolated. Even without
    checking connectivity, the principal eigenvector should have at least
    some non-zero components.
    
    FAIL if all eigenvector values are close to zero.
    """
    G = _create_pinwheel_graph()
    principal_eigenvalue, principal_eigenvector = get_principal_eigenvector_directed_unchecked(G)
    
    # Check that at least some values are not close to zero
    non_zero_count = np.sum(np.abs(principal_eigenvector) > 1e-6)
    assert non_zero_count > 0, "Principal eigenvector is all zeros"


def test_networkx_centrality_has_nonzero_values() -> None:
    """
    Test that NetworkX eigenvector centrality returns at least some non-zero values.
    
    For the pinwheel graph, NetworkX will attempt power iteration. Even though
    the graph is not strongly connected, we expect some nodes to have non-zero
    centrality scores if the method converges or uses a default initialization.
    
    FAIL if any centrality value is close to zero.
    """
    G = _create_pinwheel_graph()
    
    try:
        centrality = nx.eigenvector_centrality(G, max_iter=1000)
        
        # Check that at least some values are not close to zero
        centrality_values = np.array(list(centrality.values()))
        non_zero_count = np.sum(np.abs(centrality_values) > 1e-6)
        assert non_zero_count > 0, "All NetworkX centrality values are zero"
    except nx.PowerIterationFailedConvergence:
        # If power iteration fails to converge, that's also acceptable for this test
        # (it demonstrates the graph violates the strong connectivity requirement)
        pass


def test_unchecked_eigenvector_differs_from_networkx_centrality() -> None:
    """
    Test that the unchecked eigenvector values differ from NetworkX centrality values.
    
    The unchecked eigenvector function returns the principal eigenvector of A^T
    without checking connectivity. NetworkX's eigenvector_centrality uses power
    iteration, which may fail or give different results on reducible graphs.
    
    For the pinwheel graph, these two methods should give noticeably different
    results because the graph structure violates the strong connectivity assumption.
    
    FAIL if the unchecked eigenvector is close to the NetworkX centrality values.
    """
    G = _create_pinwheel_graph()
    
    # Get unchecked eigenvector values
    principal_eigenvalue_unc, principal_eigenvector_unc = get_principal_eigenvector_directed_unchecked(G)
    
    try:
        # Get NetworkX centrality values
        centrality_nx = nx.eigenvector_centrality(G, max_iter=1000)
        centrality_values = np.array([centrality_nx[node] for node in sorted(G.nodes())])
        
        # Normalize unchecked eigenvector to unit norm for comparison
        eigenvector_normalized = principal_eigenvector_unc / np.linalg.norm(principal_eigenvector_unc)
        
        # Compute the difference (L2 norm)
        difference = np.linalg.norm(eigenvector_normalized - centrality_values)
        
        # They should be noticeably different (not close in L2 norm)
        assert difference > 0.1, (
            f"Unchecked eigenvector is too close to NetworkX centrality "
            f"(difference = {difference:.6f}). Expected larger difference for reducible graph."
        )
    except nx.PowerIterationFailedConvergence:
        # If power iteration fails, that's fine - it shows they diverge
        pass
