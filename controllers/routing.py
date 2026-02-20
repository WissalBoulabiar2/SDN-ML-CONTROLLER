"""
Routing Engine - Dijkstra shortest path + ML-enhanced link weight prediction
"""

import logging
import numpy as np
import networkx as nx
import joblib
import os

logger = logging.getLogger(__name__)


class RoutingEngine:
    """
    Hybrid routing engine combining:
    - Dijkstra shortest path on weighted graph
    - ML model for link failure/cost prediction (RandomForest)
    """

    def __init__(self, network_graph):
        self.network_graph = network_graph
        self.ml_model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'ml_models', 'routing_model.pkl'
        )
        if os.path.exists(model_path):
            try:
                self.ml_model = joblib.load(model_path)
                logger.info("[Routing] ML routing model loaded")
            except Exception as e:
                logger.warning(f"[Routing] Could not load ML model: {e}")
        else:
            logger.info("[Routing] No ML model found - using pure Dijkstra")

    def get_path(self, src_dpid, dst_dpid):
        """
        Get best path from src to dst switch.
        Uses ML-predicted weights if model available, else hop count.
        Returns list of dpids.
        """
        graph = self.network_graph.graph
        if src_dpid not in graph or dst_dpid not in graph:
            return None

        if src_dpid == dst_dpid:
            return [src_dpid]

        try:
            if self.ml_model:
                self._update_ml_weights(graph)
            path = nx.dijkstra_path(graph, src_dpid, dst_dpid, weight='weight')
            logger.debug(f"[Routing] Path {src_dpid}→{dst_dpid}: {path}")
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"[Routing] No path: {src_dpid} → {dst_dpid}")
            return None
        except Exception as e:
            logger.error(f"[Routing] Path error: {e}")
            return None

    def _update_ml_weights(self, graph):
        """Use ML model to predict link costs and update graph weights."""
        for u, v, data in graph.edges(data=True):
            features = self._extract_link_features(u, v, data)
            if features is not None:
                try:
                    predicted_cost = self.ml_model.predict([features])[0]
                    graph[u][v]['weight'] = max(1.0, predicted_cost)
                except Exception:
                    pass  # Fall back to existing weight

    def _extract_link_features(self, src, dst, edge_data):
        """Extract feature vector for a link."""
        try:
            return [
                edge_data.get('tx_bytes', 0),
                edge_data.get('rx_bytes', 0),
                edge_data.get('tx_packets', 0),
                edge_data.get('rx_packets', 0),
                edge_data.get('tx_errors', 0),
                edge_data.get('rx_errors', 0),
                edge_data.get('latency', 1.0),
                edge_data.get('bandwidth', 100.0),
                edge_data.get('utilization', 0.0),
            ]
        except Exception:
            return None

    def get_all_paths(self, src_dpid, dst_dpid, k=3):
        """Get k shortest paths for load balancing."""
        graph = self.network_graph.graph
        try:
            paths = list(nx.shortest_simple_paths(graph, src_dpid, dst_dpid, weight='weight'))
            return paths[:k]
        except Exception:
            return []

    def recompute_all_paths(self):
        """Force recomputation of all paths (call after topology change)."""
        logger.info("[Routing] Recomputing all paths after topology change")
        # NetworkX recomputes lazily - just log
        return True

