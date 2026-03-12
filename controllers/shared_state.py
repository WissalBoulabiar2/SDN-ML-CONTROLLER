"""
shared_state.py — État partagé entre contrôleurs Ryu
=====================================================
Utilisé par stp_core.py, dhcp_controller.py, rl_controller.py
si déployés ensemble. En mode standalone (stp_core.py seul),
cet import est optionnel.
"""

from ryu.base import app_manager
import networkx as nx
from collections import defaultdict


class SharedState(app_manager.RyuApp):
    """Contexte partagé injecté dans tous les contrôleurs."""
    _CONTEXTS = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.datapaths   = {}                  # dpid → datapath
        self.mac_to_port = defaultdict(dict)   # dpid → {mac: port}
        self.graph       = nx.Graph()          # Graphe topologie (pour RL)
