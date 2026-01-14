import pytest
import networkx as nx
import numpy as np
from centrality_utilities import (
    get_principal_eigenvector_undirected,
    get_principal_eigenvector_directed
)


class TestGetPrincipalEigenvectorUndirected:
    """Test suite for get_principal_eigenvector_undirected function."""
    
    def test_connected_graph_returns_valid_results(self) -> None:
        """Test that a connected undirected graph returns valid eigenvalue and eigenvector."""
        # Create a simple connected graph
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # Check that eigenvalue is real and positive
        assert np.isreal(eigenvalue)
        assert eigenvalue > 0
        
        # Check that eigenvector has correct shape
        assert eigenvector.shape == (len(G.nodes),)
        
        # Check that eigenvector is not all zeros
        assert not np.allclose(eigenvector, 0)
    
    def test_cycle_graph(self) -> None:
        """Test with a cycle graph where all nodes should have equal centrality."""
        G = nx.cycle_graph(5)
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # For a cycle graph, the principal eigenvalue should be 2
        assert np.isclose(eigenvalue, 2.0, atol=1e-10)
        
        # All eigenvector components should have equal magnitude
        abs_values = np.abs(eigenvector)
        assert np.allclose(abs_values, abs_values[0], atol=1e-10)
    
    def test_complete_graph(self) -> None:
        """Test with a complete graph."""
        G = nx.complete_graph(5)
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # For complete graph K_n, principal eigenvalue should be n-1
        assert np.isclose(eigenvalue, 4.0, atol=1e-10)
        
        # All components should have equal magnitude (up to sign)
        abs_values = np.abs(eigenvector)
        assert np.allclose(abs_values, abs_values[0], atol=1e-10)
    
    def test_raises_error_for_directed_graph(self) -> None:
        """Test that passing a directed graph raises TypeError."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        with pytest.raises(TypeError, match="Graph must be undirected"):
            get_principal_eigenvector_undirected(G)
    
    def test_raises_error_for_disconnected_graph(self) -> None:
        """Test that passing a disconnected graph raises ValueError."""
        G = nx.Graph()
        # Create two disconnected components
        G.add_edges_from([(1, 2), (2, 3)])
        G.add_edges_from([(4, 5), (5, 6)])
        
        with pytest.raises(ValueError, match="must be connected"):
            get_principal_eigenvector_undirected(G)
    
    def test_single_node_graph(self) -> None:
        """Test with a single node (trivially connected)."""
        G = nx.Graph()
        G.add_node(1)
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # Single node has eigenvalue 0
        assert np.isclose(eigenvalue, 0.0, atol=1e-10)
        assert len(eigenvector) == 1
    
    def test_karate_club_graph(self) -> None:
        """Test with the well-known Karate Club graph."""
        G = nx.karate_club_graph()
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # Check basic properties
        assert eigenvalue > 0
        assert eigenvector.shape == (34,)  # Karate club has 34 nodes
        
        # The principal eigenvalue should be largest in magnitude
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray()
        all_eigenvalues = np.linalg.eigvals(A)
        assert np.isclose(eigenvalue, max(np.abs(all_eigenvalues)), atol=1e-10)


class TestGetPrincipalEigenvectorDirected:
    """Test suite for get_principal_eigenvector_directed function."""
    
    def test_strongly_connected_graph_returns_valid_results(self) -> None:
        """Test that a strongly connected directed graph returns valid eigenvalue and eigenvector."""
        # Create a simple strongly connected directed graph (cycle)
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # Check that eigenvalue is real and positive
        assert np.isreal(eigenvalue)
        assert np.real(eigenvalue) > 0
        
        # Check that eigenvector has correct shape
        assert eigenvector.shape == (len(G.nodes),)
        
        # Check that eigenvector is not all zeros
        assert not np.allclose(eigenvector, 0)
    
    def test_directed_cycle(self) -> None:
        """Test with a directed cycle graph."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # For a directed cycle, principal eigenvalue should be 1
        assert np.isclose(np.abs(eigenvalue), 1.0, atol=1e-10)
        
        # All eigenvector components should have equal magnitude
        abs_values = np.abs(eigenvector)
        assert np.allclose(abs_values, abs_values[0], atol=1e-10)
    
    def test_complete_directed_graph(self) -> None:
        """Test with a complete directed graph (tournament)."""
        # Create a strongly connected directed graph
        G = nx.DiGraph()
        nodes = [1, 2, 3, 4]
        # Add edges in both directions for all pairs
        for i in nodes:
            for j in nodes:
                if i != j:
                    G.add_edge(i, j)
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # Principal eigenvalue should be n-1 for complete directed graph
        assert np.isclose(eigenvalue, 3.0, atol=1e-10)
        
        # All components should have equal magnitude
        abs_values = np.abs(eigenvector)
        assert np.allclose(abs_values, abs_values[0], atol=1e-10)
    
    def test_raises_error_for_undirected_graph(self) -> None:
        """Test that passing an undirected graph raises TypeError."""
        G = nx.Graph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1)])
        
        with pytest.raises(TypeError, match="must be a directed graph"):
            get_principal_eigenvector_directed(G)
    
    def test_raises_error_for_weakly_connected_graph(self) -> None:
        """Test that passing a weakly (not strongly) connected graph raises ValueError."""
        G = nx.DiGraph()
        # Create a directed path (weakly but not strongly connected)
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        
        with pytest.raises(ValueError, match="must be strongly connected"):
            get_principal_eigenvector_directed(G)
    
    def test_raises_error_for_graph_with_dead_end(self) -> None:
        """Test that a graph with a dead-end node (no outgoing edges) raises ValueError."""
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])  # 4 is a dead-end
        
        with pytest.raises(ValueError, match="must be strongly connected"):
            get_principal_eigenvector_directed(G)
    
    def test_raises_error_for_graph_with_spider_trap(self) -> None:
        """Test that a graph with a spider trap raises ValueError."""
        G = nx.DiGraph()
        # Main component
        G.add_edges_from([(1, 2), (2, 3), (3, 4)])
        # Spider trap (can enter but not leave)
        G.add_edges_from([(4, 5), (5, 6), (6, 5)])
        
        with pytest.raises(ValueError, match="must be strongly connected"):
            get_principal_eigenvector_directed(G)
    
    def test_strongly_connected_with_extra_edges(self) -> None:
        """Test a strongly connected graph with various edge configurations."""
        G = nx.DiGraph()
        # Create base cycle for strong connectivity
        G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)])
        # Add extra edges
        G.add_edges_from([(1, 3), (2, 4), (3, 5)])
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # Should successfully compute without errors
        assert eigenvalue > 0
        assert eigenvector.shape == (5,)
    
    def test_single_node_graph(self) -> None:
        """Test with a single node (trivially strongly connected)."""
        G = nx.DiGraph()
        G.add_node(1)
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # Single node has eigenvalue 0
        assert np.isclose(eigenvalue, 0.0, atol=1e-10)
        assert len(eigenvector) == 1


class TestEigenvalueComparison:
    """Test that eigenvalues are computed using magnitude correctly."""
    
    def test_undirected_uses_magnitude(self) -> None:
        """Test that undirected function correctly identifies largest eigenvalue by magnitude."""
        # Create a bipartite graph which might have negative eigenvalues
        G = nx.complete_bipartite_graph(3, 3)
        
        eigenvalue, eigenvector = get_principal_eigenvector_undirected(G)
        
        # Verify this is indeed the largest by magnitude
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray()
        all_eigenvalues = np.linalg.eigvals(A)
        max_magnitude = np.max(np.abs(all_eigenvalues))
        
        assert np.isclose(np.abs(eigenvalue), max_magnitude, atol=1e-10)
    
    def test_directed_uses_magnitude(self) -> None:
        """Test that directed function correctly identifies largest eigenvalue by magnitude."""
        # Create a strongly connected directed graph
        G = nx.DiGraph()
        G.add_edges_from([(1, 2), (2, 3), (3, 1), (1, 3), (2, 1)])
        
        eigenvalue, eigenvector = get_principal_eigenvector_directed(G)
        
        # Verify this is indeed the largest by magnitude
        A = nx.adjacency_matrix(G, nodelist=sorted(G.nodes())).toarray()
        all_eigenvalues = np.linalg.eigvals(A.T)
        max_magnitude = np.max(np.abs(all_eigenvalues))
        
        assert np.isclose(np.abs(eigenvalue), max_magnitude, atol=1e-10)
