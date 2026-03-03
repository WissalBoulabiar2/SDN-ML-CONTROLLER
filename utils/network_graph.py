"""
Network Graph - SDN Topology Maintenance
Maintains a weighted NetworkX graph of the network topology.
Updates link weights from port statistics for ML-enhanced routing.
"""

import logging
import networkx as nx
import time
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger(__name__)


class NetworkGraph:
    """
    Live network topology graph using NetworkX.
    - Nodes: switches (dpid)
    - Edges: links with weight attributes (latency, utilization, errors)
    - Updated in real-time from Ryu topology events and port stats
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self.port_map: Dict[Tuple[int, int], int] = {}  # {(src_dpid, dst_dpid): src_port}
        self.last_update = time.time()
        self.port_stats_history: Dict[int, list] = {}  # {dpid: [stats]}
        self.switches: Dict[int, dict] = {}
        self.links: Dict[Tuple[int, int], dict] = {}

    # ─── Topology Management ─────────────────────────────────────────

    def add_switch(self, dpid: int, **attrs):
        """Add a switch node to the graph."""
        self.graph.add_node(dpid, **attrs)
        self.switches[dpid] = {'dpid': dpid, 'added_at': time.time(), **attrs}
        logger.debug(f"[NetworkGraph] Switch added: {dpid}")

    def remove_switch(self, dpid: int):
        """Remove a switch and all its links."""
        if dpid in self.graph:
            self.graph.remove_node(dpid)
        self.switches.pop(dpid, None)
        # Remove port map entries
        self.port_map = {k: v for k, v in self.port_map.items()
                         if k[0] != dpid and k[1] != dpid}
        logger.info(f"[NetworkGraph] Switch removed: {dpid}")

    def add_link(self, src_dpid: int, dst_dpid: int,
                 src_port: int, dst_port: int, **attrs):
        """Add a directed link between two switches."""
        if src_dpid not in self.graph:
            self.add_switch(src_dpid)
        if dst_dpid not in self.graph:
            self.add_switch(dst_dpid)

        default_attrs = {
            'src_port': src_port,
            'dst_port': dst_port,
            'weight': 1.0,
            'bandwidth': 100.0,  # Mbps (default)
            'latency': 1.0,
            'utilization': 0.0,
            'tx_bytes': 0,
            'rx_bytes': 0,
            'tx_packets': 0,
            'rx_packets': 0,
            'tx_errors': 0,
            'rx_errors': 0,
            'added_at': time.time()
        }
        default_attrs.update(attrs)

        self.graph.add_edge(src_dpid, dst_dpid, **default_attrs)
        self.port_map[(src_dpid, dst_dpid)] = src_port
        self.links[(src_dpid, dst_dpid)] = default_attrs
        logger.debug(f"[NetworkGraph] Link added: {src_dpid}:{src_port} → {dst_dpid}:{dst_port}")

    def remove_link(self, src_dpid: int, dst_dpid: int):
        """Remove a link from the graph."""
        if self.graph.has_edge(src_dpid, dst_dpid):
            self.graph.remove_edge(src_dpid, dst_dpid)
        self.port_map.pop((src_dpid, dst_dpid), None)
        self.links.pop((src_dpid, dst_dpid), None)
        logger.info(f"[NetworkGraph] Link removed: {src_dpid} → {dst_dpid}")

    def get_port(self, src_dpid: int, dst_dpid: int) -> Optional[int]:
        """Get the output port on src_dpid to reach dst_dpid."""
        return self.port_map.get((src_dpid, dst_dpid))

    # ─── Statistics Updates ──────────────────────────────────────────

    def update_port_stats(self, dpid: int, port_stats: list):
        """
        Update link weights from port statistics.
        Called from PortStatsReply handler.
        """
        self.port_stats_history.setdefault(dpid, [])
        prev_stats = {s['port_no']: s for s in self.port_stats_history.get(dpid, [])}

        for stat in port_stats:
            port_no = stat.port_no
            if port_no == 0xffffffff:  # LOCAL port
                continue

            # Compute deltas
            tx_bytes = stat.tx_bytes
            rx_bytes = stat.rx_bytes
            tx_errors = stat.tx_errors
            rx_errors = stat.rx_errors

            # Find the link using this port
            for (src, dst), port in self.port_map.items():
                if src == dpid and port == port_no:
                    if self.graph.has_edge(src, dst):
                        edge_data = self.graph[src][dst]
                        dt = max(1, time.time() - edge_data.get('added_at', time.time()))

                        # Update bytes/packets
                        edge_data['tx_bytes'] = tx_bytes
                        edge_data['rx_bytes'] = rx_bytes
                        edge_data['tx_errors'] = tx_errors
                        edge_data['rx_errors'] = rx_errors

                        # Compute utilization (fraction of 100Mbps link)
                        bandwidth_bps = edge_data.get('bandwidth', 100.0) * 1e6
                        tx_rate = tx_bytes * 8 / dt
                        utilization = min(1.0, tx_rate / bandwidth_bps)
                        edge_data['utilization'] = round(utilization, 4)

                        # Update composite weight:
                        # weight = base + utilization_penalty + error_penalty
                        error_rate = tx_errors / max(1, stat.tx_packets)
                        edge_data['weight'] = (
                            1.0 +
                            utilization * 10.0 +          # Heavy utilization penalty
                            error_rate * 5.0               # Error penalty
                        )
                        break

        # Store current stats for next delta
        self.port_stats_history[dpid] = [
            {'port_no': s.port_no, 'tx_bytes': s.tx_bytes, 'rx_bytes': s.rx_bytes,
             'tx_errors': s.tx_errors, 'rx_errors': s.rx_errors,
             'tx_packets': s.tx_packets}
            for s in port_stats
        ]
        self.last_update = time.time()

    def update_from_ryu(self, switches: list, links: list):
        """
        Full topology refresh from Ryu topology API.
        Called periodically from the topology thread.
        """
        # Add any new switches
        for sw in switches:
            if sw.dp.id not in self.graph:
                self.add_switch(sw.dp.id)

        # Add any new links
        existing_links = set(self.graph.edges())
        for link in links:
            src, dst = link.src.dpid, link.dst.dpid
            if (src, dst) not in existing_links:
                self.add_link(src, dst, link.src.port_no, link.dst.port_no)

    # ─── Query Methods ───────────────────────────────────────────────

    def get_shortest_path(self, src: int, dst: int) -> Optional[List[int]]:
        """Get shortest path by weight."""
        try:
            return nx.dijkstra_path(self.graph, src, dst, weight='weight')
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_link_utilization(self) -> Dict[Tuple[int, int], float]:
        """Get utilization of all links."""
        return {
            (u, v): data.get('utilization', 0.0)
            for u, v, data in self.graph.edges(data=True)
        }

    def get_topology_dict(self) -> dict:
        """Serialize topology to JSON-compatible dict."""
        nodes = [{'id': n, **self.switches.get(n, {})} for n in self.graph.nodes()]
        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'src': u, 'dst': v,
                'src_port': data.get('src_port'),
                'dst_port': data.get('dst_port'),
                'weight': round(data.get('weight', 1.0), 4),
                'utilization': round(data.get('utilization', 0.0), 4),
                'tx_bytes': data.get('tx_bytes', 0),
                'rx_bytes': data.get('rx_bytes', 0),
            })
        return {
            'nodes': nodes,
            'edges': edges,
            'switch_count': len(nodes),
            'link_count': len(edges),
            'last_update': self.last_update
        }

    def is_connected(self) -> bool:
        """Check if the network graph is fully connected."""
        if len(self.graph.nodes()) < 2:
            return True
        return nx.is_weakly_connected(self.graph)

    def get_link_features(self, src: int, dst: int) -> Optional[dict]:
        """Get all features for a specific link."""
        if self.graph.has_edge(src, dst):
            return dict(self.graph[src][dst])
        return None

    # ─── DDPG Reward — M/M/1/K (Kim et al., IEEE Access 2022) ───────

    def compute_reward(self, paths: list,
                       lambda_per_switch: dict,
                       alpha: float = 0.9,
                       mu: float = 3000.0,
                       K: int = 10_000) -> float:
        """
        Calcule le reward DDPG selon eq. 14 du paper:
          R = α × rd + (1-α) × rp

        rd = 1 - D_avg / D_max   (délai normalisé, eq. 12)
        rp = 1 - L_tot / Σλn     (perte normalisée, eq. 13)

        Utilise le modèle M/M/1/K pour calculer délai et perte.

        Arguments:
          paths             : liste des chemins actifs [[s1,s2,...], ...]
          lambda_per_switch : {dpid: taux_arrivée pkt/s}
          alpha             : poids délai vs perte (paper: 0.9)
          mu                : service rate (paper: 3000 pkt/s)
          K                 : capacité système (paper: 10,000)
        """
        import numpy as np

        def mm1k_metrics(lam_n):
            """Délai et prob. perte pour un switch M/M/1/K."""
            if lam_n <= 0:
                return 0.0, 0.0
            rho = lam_n / mu
            if abs(rho - 1.0) < 1e-9:
                Pb  = 1.0 / (K + 1.0)
                E_N = K / 2.0
            else:
                Pb  = (1 - rho) * rho**K / (1 - rho**(K + 1) + 1e-12)
                E_N = rho / (1 - rho) - \
                      (K + 1) * rho**(K + 1) / (1 - rho**(K + 1) + 1e-12)
            E_N = max(0.0, E_N)
            Pb  = float(np.clip(Pb, 0.0, 1.0))
            eff = lam_n * (1 - Pb)
            delay = E_N / eff if eff > 1e-6 else 0.0
            return delay, Pb

        # Délai end-to-end moyen (eq. 4-5)
        total_delay = 0.0
        n_flows     = max(len(paths), 1)
        for path in paths:
            for sw in path:
                lam_n = lambda_per_switch.get(sw, 10.0)
                d, _  = mm1k_metrics(lam_n)
                total_delay += d
        D_avg = total_delay / n_flows

        # Perte totale (eq. 7)
        L_tot     = 0.0
        lam_total = 0.0
        for sw, lam_n in lambda_per_switch.items():
            _, Pb  = mm1k_metrics(lam_n)
            L_tot     += lam_n * Pb
            lam_total += lam_n

        # D_max (eq. 12) = délai max théorique sur chemin le plus long
        N_switches = len(self.graph.nodes())
        D_max      = N_switches * (K / mu)

        rd = 1.0 - float(np.clip(D_avg / (D_max + 1e-9), 0.0, 1.0))
        rp = 1.0 - float(np.clip(L_tot / (lam_total + 1e-9), 0.0, 1.0))

        return float(np.clip(alpha * rd + (1 - alpha) * rp, 0.0, 1.0))

    def __len__(self):
        return len(self.graph.nodes())

    def __repr__(self):
        return (f"NetworkGraph(switches={len(self.graph.nodes())}, "
                f"links={len(self.graph.edges())})")
