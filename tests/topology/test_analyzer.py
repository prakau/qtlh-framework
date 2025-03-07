"""
Tests for the topological analysis module.
"""

import numpy as np
import pytest
import networkx as nx
from qtlh.topology import TopologicalAnalyzer, TopologyConfig
from qtlh.topology.analyzer import PersistentHomology, MapperAlgorithm

@pytest.fixture
def point_cloud():
    """Generate test point cloud data."""
    np.random.seed(42)
    # Generate points in a circle with some noise
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    r = 1 + 0.1 * np.random.randn(n_points)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])

@pytest.mark.topology
def test_topology_config():
    """Test topology configuration initialization."""
    config = TopologyConfig(
        max_dimension=3,
        max_scale=2.0,
        n_bins=50,
        overlap_fraction=0.3
    )
    assert config.max_dimension == 3
    assert config.max_scale == 2.0
    assert config.n_bins == 50
    assert config.overlap_fraction == 0.3

@pytest.mark.topology
def test_persistent_homology_initialization(topology_config):
    """Test persistent homology analyzer initialization."""
    ph = PersistentHomology(topology_config)
    assert ph.config == topology_config
    assert hasattr(ph, 'persistence')

@pytest.mark.topology
def test_mapper_algorithm_initialization(topology_config):
    """Test mapper algorithm initialization."""
    mapper = MapperAlgorithm(topology_config)
    assert mapper.config == topology_config

@pytest.mark.topology
def test_persistent_homology_computation(topo_analyzer, point_cloud):
    """Test persistent homology computation."""
    results = topo_analyzer.persistence.compute_persistence(point_cloud)
    
    # Check results structure
    assert 'persistence_diagrams' in results
    assert 'betti_curves' in results
    assert 'persistence_entropy' in results
    
    # Check persistence diagrams
    diagrams = results['persistence_diagrams']
    assert isinstance(diagrams, list)
    for dim_id, (birth, death) in diagrams:
        assert isinstance(dim_id, int)
        assert birth <= death

@pytest.mark.topology
def test_mapper_graph_construction(topo_analyzer, point_cloud):
    """Test mapper graph construction."""
    # Construct mapper graph
    graph = topo_analyzer.mapper.construct_mapper_graph(
        point_cloud,
        filter_function=lambda x: np.sum(x**2, axis=1)
    )
    
    # Check graph properties
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    
    # Check node attributes
    for node in graph.nodes():
        assert 'points' in graph.nodes[node]
        assert isinstance(graph.nodes[node]['points'], np.ndarray)

@pytest.mark.topology
def test_sequence_analysis(topo_analyzer, mock_features):
    """Test full sequence analysis pipeline."""
    results = topo_analyzer.analyze_sequence(
        mock_features['quantum'],
        compute_mapper=True
    )
    
    # Check results structure
    assert 'persistence_results' in results
    assert 'topological_summary' in results
    assert 'mapper_graph' in results
    assert 'graph_properties' in results

@pytest.mark.topology
def test_betti_curves(topo_analyzer, point_cloud):
    """Test Betti curves computation."""
    results = topo_analyzer.persistence.compute_persistence(point_cloud)
    betti_curves = results['betti_curves']
    
    assert isinstance(betti_curves, np.ndarray)
    assert betti_curves.ndim == 2
    assert betti_curves.shape[0] <= topo_analyzer.config.max_dimension + 1

@pytest.mark.topology
def test_persistence_statistics(topo_analyzer, point_cloud):
    """Test persistence statistics computation."""
    results = topo_analyzer.persistence.compute_persistence(point_cloud)
    stats = topo_analyzer.persistence._compute_persistence_stats(results['persistence_diagrams'])
    
    assert isinstance(stats, dict)
    for dim in range(topo_analyzer.config.max_dimension + 1):
        key = f'dim_{dim}'
        assert key in stats
        assert isinstance(stats[key], np.ndarray)
        assert stats[key].shape == (5,)  # mean, std, max, sum, count

@pytest.mark.topology
def test_mapper_cover_creation(topology_config):
    """Test cover creation for mapper algorithm."""
    mapper = MapperAlgorithm(topology_config)
    values = np.linspace(0, 1, 100)
    intervals = mapper._create_cover(values)
    
    assert isinstance(intervals, list)
    assert len(intervals) == topology_config.mapper_resolution
    for start, end in intervals:
        assert start < end

@pytest.mark.topology
def test_cluster_overlaps(topology_config):
    """Test cluster overlap detection."""
    mapper = MapperAlgorithm(topology_config)
    
    # Create two overlapping clusters
    cluster1 = np.array([[0, 0], [1, 1]])
    cluster2 = np.array([[0.5, 0.5], [1.5, 1.5]])
    
    assert mapper._clusters_overlap(cluster1, cluster2)
    
    # Create non-overlapping clusters
    cluster3 = np.array([[10, 10], [11, 11]])
    assert not mapper._clusters_overlap(cluster1, cluster3)

@pytest.mark.topology
def test_topological_summary(topo_analyzer, point_cloud):
    """Test topological summary computation."""
    results = topo_analyzer.persistence.compute_persistence(point_cloud)
    summary = topo_analyzer._compute_topological_summary(results)
    
    assert isinstance(summary, dict)
    assert 'total_persistence' in summary
    assert 'persistence_entropy' in summary
    assert 'betti_numbers' in summary

@pytest.mark.topology
def test_graph_analysis(topo_analyzer, point_cloud):
    """Test graph property analysis."""
    graph = topo_analyzer.mapper.construct_mapper_graph(point_cloud)
    properties = topo_analyzer._analyze_graph(graph)
    
    assert isinstance(properties, dict)
    assert 'n_vertices' in properties
    assert 'n_edges' in properties
    assert 'average_degree' in properties
    assert 'clustering_coefficient' in properties
    assert 'connected_components' in properties

@pytest.mark.topology
def test_save_load_results(topo_analyzer, point_cloud, tmp_path):
    """Test saving and loading analysis results."""
    # Perform analysis
    results = topo_analyzer.analyze_sequence(point_cloud)
    
    # Save results
    save_path = tmp_path / "topo_results.pkl"
    topo_analyzer.save_results(results, save_path)
    
    # Load results
    loaded_results = topo_analyzer.load_results(save_path)
    
    # Check consistency
    assert set(results.keys()) == set(loaded_results.keys())
    for key in results:
        if isinstance(results[key], np.ndarray):
            assert np.allclose(results[key], loaded_results[key])
        else:
            assert results[key] == loaded_results[key]

@pytest.mark.topology
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_different_dimensions(topology_config, point_cloud, dimension):
    """Test analysis with different maximum homology dimensions."""
    config = TopologyConfig(**{**vars(topology_config), 'max_dimension': dimension})
    analyzer = TopologicalAnalyzer(config)
    
    results = analyzer.analyze_sequence(point_cloud)
    persistence_results = results['persistence_results']
    
    # Check maximum homology dimension
    max_dim = max(dim for dim, _ in persistence_results['persistence_diagrams'])
    assert max_dim <= dimension
