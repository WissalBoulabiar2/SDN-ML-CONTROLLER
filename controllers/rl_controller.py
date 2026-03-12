"""
rl_controller.py — RL Routing + MAC Forwarding (module isolé)
=============================================================
Fixes appliqués :
  3. Modularisé : SharedState injecté, plus couplé à stp_core
  2. Graphe NetworkX mis à jour via SharedState (partagé avec stp_core LLDP)

Responsabilités :
  - Forwarding MAC classique (mac_to_port)
  - RL path selection via QRoutingAgent (si dst_dpid connu)
  - Installation des flow entries (priorité 1)
  - Ne touche PAS aux BPDUs ni au DHCP (délégués à leurs modules)

Usage :
  PYTHONPATH=. ryu-manager \\
    controllers/shared_state.py \\
    controllers/stp_core.py \\
    controllers/dhcp_controller.py \\
    controllers/rl_controller.py \\
    --observe-links
"""

import logging

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import stplib

from controllers.shared_state import SharedState
from controllers.sdn_rl_routing import QRoutingAgent, SDNNetwork

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BPDU_DST_STD  = '01:80:c2:00:00:00'
BPDU_DST_PVST = '01:00:0c:cc:cc:cd'


class RLController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {
        'shared_state': SharedState,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shared           = kwargs['shared_state']
        self.datapaths   = shared.datapaths
        self.mac_to_port = shared.mac_to_port
        self.rl_graph    = shared.graph  # graphe mis à jour par stp_core via LLDP

        # RL agent — graphe partagé, pas de copie
        self.rl_net   = SDNNetwork(self.rl_graph, mu=600, K=20, seed=None)
        self.rl_agent = QRoutingAgent(self.rl_graph, self.rl_net)
        try:
            self.rl_agent.load("controllers/qtable_geant.pkl")
            logger.info("[RL] Q-table chargée depuis qtable_geant.pkl")
        except FileNotFoundError:
            logger.warning("[RL] qtable_geant.pkl introuvable — agent initialisé à zéro")

        logger.info("=" * 60)
        logger.info("  [RL] QRouting Agent + MAC Forwarding")
        logger.info("  Graphe topologique : mis à jour par LLDP (stp_core)")
        logger.info("  Fallback : OFPP_FLOOD si dst inconnu ou path vide")
        logger.info("=" * 60)

    # ── Packet in ─────────────────────────────────────────────────────────────

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        ofproto  = datapath.ofproto
        parser   = datapath.ofproto_parser
        in_port  = msg.match['in_port']
        dpid     = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore les BPDUs (traités par stp_core) et DHCP (traités par dhcp_controller)
        if eth.dst.lower() in (BPDU_DST_STD, BPDU_DST_PVST):
            return

        # ── Apprentissage MAC ─────────────────────────────────────────────────
        if dpid not in self.mac_to_port:
            self.mac_to_port[dpid] = {}
        self.mac_to_port[dpid][eth.src] = in_port

        # ── Décision de forwarding ────────────────────────────────────────────
        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)

        if out_port == ofproto.OFPP_FLOOD:
            out_port = self._rl_lookup(dpid, eth.dst, ofproto)

        actions = [parser.OFPActionOutput(out_port)]

        # Installe un flow uniquement si la destination est connue
        if out_port != ofproto.OFPP_FLOOD:
            self._add_flow(
                datapath, 1,
                parser.OFPMatch(in_port=in_port, eth_dst=eth.dst),
                actions
            )

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data
        ))

    # ── RL path lookup ────────────────────────────────────────────────────────

    def _rl_lookup(self, src_dpid, dst_mac, ofproto):
        """
        Cherche le dpid destination via mac_to_port, puis calcule le chemin RL.
        Retourne le port de sortie local vers le prochain saut, ou OFPP_FLOOD.
        """
        dst_dpid = None
        for sw, macs in self.mac_to_port.items():
            if sw != src_dpid and dst_mac in macs:
                dst_dpid = sw
                break

        if dst_dpid is None:
            return ofproto.OFPP_FLOOD

        try:
            path = self.rl_agent.get_path(src_dpid, dst_dpid, lam=200, fallback=None)
        except Exception as e:
            logger.debug(f"[RL] get_path error: {e}")
            return ofproto.OFPP_FLOOD

        if not path or len(path) < 2:
            return ofproto.OFPP_FLOOD

        next_hop = path[1]
        # Cherche le port local qui mène vers next_hop
        learned_at_next = self.mac_to_port.get(next_hop, {})
        for mac, port in self.mac_to_port[src_dpid].items():
            if mac in learned_at_next:
                logger.info(
                    f"[RL] path {src_dpid}→{dst_dpid}: {path} → out_port={port}"
                )
                return port

        return ofproto.OFPP_FLOOD

    # ── Flow helper ───────────────────────────────────────────────────────────

    def _add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst    = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod     = parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            match=match, instructions=inst
        )
        datapath.send_msg(mod)
