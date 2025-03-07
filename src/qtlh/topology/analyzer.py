"""
Topological Analysis Module
=========================

This module implements advanced topological data analysis techniques for genomic sequences,
including persistent homology, mapper algorithm, and zigzag persistence.
"""

import numpy as np
import gudhi as gd
import dionysus as d
from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import PersistenceEntropy, BettiCurve
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import networkx as nx
from scipy.spatial.distance import cdist
import logging

@dataclass
class TopologyConfig:
    """Configuration for topological analysis."""
    max_dimension: int = 3
    max_scale: float = 2.0
    n_bins: int = 50
    overlap_fraction: float = 0.3
    min_persistence: float = 0.1
    mapper_resolution: int = 10
    mapper_gain: float = 1.0

class PersistentHomology:
    """Implements persistent homology computation."""
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        self.persistence = VietorisRipsPersistence(
            metric="euclidean",
            homology_dimensions=range(config.max_dimension + 1)
        )
        
    def compute_persistence(
        self,
        point_cloud: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Compute persistent homology using multiple approaches."""
        # Compute using GUDHI
        simplex_tree = gd.SimplexTree()
        rips = gd.RipsComplex(points=point_cloud, max_edge_length=self.config.max_scale)
        simplex_tree = rips.create_simplex_tree(max_dimension=self.config.max_dimension)
        gudhi_diagrams = simplex_tree.persistence()
        
        # Compute using Dionysus
        filtration = d.fill_rips(point_cloud, self.config.max_dimension, self.config.max_scale)
        persistence = d.homology_persistence(filtration)
        dionysus_diagrams = d.init_diagrams(persistence, filtration)
        
        # Compute using GTDA
        gtda_diagrams = self.persistence.fit_transform([point_cloud])
        
        # Extract statistical features
        persistence_stats = self._compute_persistence_stats(gudhi_diagrams)
        betti_curves = self._compute_betti_curves(gtda_diagrams)
        entropy = self._compute_persistence_entropy(gtda_diagrams)
        
        return {
            'persistence_diagrams': gudhi_diagrams,
            'dionysus_diagrams': dionysus_diagrams,
            'gtda_diagrams': gtda_diagrams,
            'persistence_stats': persistence_stats,
            'betti_curves': betti_curves,
            'persistence_entropy': entropy
        }
    
    def _compute_persistence_stats(
        self,
        diagrams: List[Tuple]
    ) -> Dict[str, np.ndarray]:
        """Compute statistical features from persistence diagrams."""
        stats = {}
        
        for dim in range(self.config.max_dimension + 1):
            dim_pairs = [(birth, death) for dim_id, (birth, death) in diagrams if dim_id == dim]
            if not dim_pairs:
                stats[f'dim_{dim}'] = np.zeros(5)
                continue
                
            pairs = np.array(dim_pairs)
            lifetimes = pairs[:, 1] - pairs[:, 0]
            
            stats[f'dim_{dim}'] = np.array([
                np.mean(lifetimes),
                np.std(lifetimes),
                np.max(lifetimes),
                np.sum(lifetimes),
                len(lifetimes)
            ])
            
        return stats
    
    def _compute_betti_curves(
        self,
        diagrams: np.ndarray
    ) -> np.ndarray:
        """Compute Betti curves from persistence diagrams."""
        betti = BettiCurve()
        return betti.fit_transform(diagrams)
    
    def _compute_persistence_entropy(
        self,
        diagrams: np.ndarray
    ) -> np.ndarray:
        """Compute persistence entropy."""
        entropy = PersistenceEntropy()
        return entropy.fit_transform(diagrams)

class MapperAlgorithm:
    """Implements the Mapper algorithm for topological data visualization."""
    
    def __init__(self, config: TopologyConfig):
        self.config = config
        
    def construct_mapper_graph(
        self,
        point_cloud: np.ndarray,
        filter_function: Optional[callable] = None
    ) -> nx.Graph:
        """Construct Mapper graph from point cloud data."""
        if filter_function is None:
            filter_function = lambda x: np.sum(x**2, axis=1)
            
        # Compute filter values
        filter_values = filter_function(point_cloud)
        
        # Create cover
        intervals = self._create_cover(filter_values)
        
        # Cluster points in each interval
        vertices = []
        edges = []
        
        for interval in intervals:
            # Get points in interval
            mask = (filter_values >= interval[0]) & (filter_values <= interval[1])
            subset = point_cloud[mask]
            
            if len(subset) == 0:
                continue
                
            # Cluster points
            clusters = self._cluster_points(subset)
            
            # Add vertices
            vertices.extend(clusters)
            
            # Add edges between overlapping clusters
            edges.extend(self._find_edges(clusters))
            
        # Construct graph
        G = nx.Graph()
        
        # Add vertices with positions
        for i, cluster in enumerate(vertices):
            G.add_node(i, points=cluster)
            
        # Add edges
        for (i, j) in edges:
            G.add_edge(i, j)
            
        return G
    
    def _create_cover(
        self,
        values: np.ndarray
    ) -> List[Tuple[float, float]]:
        """Create overlapping intervals covering the range of values."""
        min_val, max_val = np.min(values), np.max(values)
        interval_size = (max_val - min_val) / self.config.mapper_resolution
        overlap_size = interval_size * self.config.overlap_fraction
        
        intervals = []
        for i in range(self.config.mapper_resolution):
            start = min_val + i * (interval_size - overlap_size)
            end = start + interval_size
            intervals.append((start, end))
            
        return intervals
    
    def _cluster_points(
        self,
        points: np.ndarray
    ) -> List[np.ndarray]:
        """Cluster points using density-based clustering."""
        from sklearn.cluster import DBSCAN
        
        # Estimate epsilon for DBSCAN
        distances = cdist(points, points)
        eps = np.mean(np.sort(distances, axis=1)[:, 1])
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=2).fit(points)
        
        # Extract clusters
        clusters = []
        for label in set(clustering.labels_):
            if label != -1:  # Exclude noise
                cluster_points = points[clustering.labels_ == label]
                clusters.append(cluster_points)
                
        return clusters
    
    def _find_edges(
        self,
        clusters: List[np.ndarray]
    ) -> List[Tuple[int, int]]:
        """Find edges between overlapping clusters."""
        edges = []
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if self._clusters_overlap(clusters[i], clusters[j]):
                    edges.append((i, j))
                    
        return edges
    
    def _clusters_overlap(
        self,
        cluster1: np.ndarray,
        cluster2: np.ndarray
    ) -> bool:
        """Check if two clusters have overlapping points."""
        distances = cdist(cluster1, cluster2)
        return np.min(distances) < self.config.min_persistence

class TopologicalAnalyzer:
    """Main class for topological analysis of genomic sequences."""
    
    def __init__(self, config: Optional[TopologyConfig] = None):
        self.config = config or TopologyConfig()
        self.persistence = PersistentHomology(self.config)
        self.mapper = MapperAlgorithm(self.config)
        
    def analyze_sequence(
        self,
        sequence_features: np.ndarray,
        compute_mapper: bool = True
    ) -> Dict:
        """Perform comprehensive topological analysis.
        
        Args:
            sequence_features: Point cloud representation of sequence
            compute_mapper: Whether to compute Mapper graph
            
        Returns:
            Dictionary containing analysis results
        """
        # Compute persistent homology
        persistence_results = self.persistence.compute_persistence(sequence_features)
        
        results = {
            'persistence_results': persistence_results,
            'topological_summary': self._compute_topological_summary(persistence_results)
        }
        
        # Optionally compute Mapper graph
        if compute_mapper:
            mapper_graph = self.mapper.construct_mapper_graph(sequence_features)
            results['mapper_graph'] = mapper_graph
            results['graph_properties'] = self._analyze_graph(mapper_graph)
            
        return results
    
    def _compute_topological_summary(
        self,
        persistence_results: Dict
    ) -> Dict[str, float]:
        """Compute summary statistics from persistence results."""
        diagrams = persistence_results['persistence_diagrams']
        
        # Compute total persistence
        total_persistence = sum(death - birth for _, (birth, death) in diagrams)
        
        # Compute persistence entropy
        entropy = persistence_results['persistence_entropy']
        
        # Compute Betti numbers
        betti_numbers = np.sum(persistence_results['betti_curves'], axis=1)
        
        return {
            'total_persistence': total_persistence,
            'persistence_entropy': float(entropy.mean()),
            'betti_numbers': betti_numbers.tolist()
        }
    
    def _analyze_graph(
        self,
        G: nx.Graph
    ) -> Dict[str, float]:
        """Analyze topological properties of Mapper graph."""
        return {
            'n_vertices': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'average_degree': np.mean([d for _, d in G.degree()]),
            'clustering_coefficient': nx.average_clustering(G),
            'connected_components': nx.number_connected_components(G)
        }
    
    def save_results(
        self,
        results: Dict,
        filepath: str
    ) -> None:
        """Save analysis results."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
    
    def load_results(
        self,
        filepath: str
    ) -> Dict:
        """Load analysis results."""
        import pickle
        with open(filepath, 'rb') as f:
            return pickle.load(f)
