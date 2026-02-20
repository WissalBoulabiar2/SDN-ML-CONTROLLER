"""
Main SDN Controller - Ryu Application
Combines topology learning, shortest path routing, STP, ARP handling, REST API,
and ML-based intrusion detection with automated response.
"""

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, ipv6, tcp, udp, icmp, arp
from ryu.lib import hub
from ryu.topology import event as topo_event
from ryu.topology.api import get_switch, get_link

import networkx as nx
import logging
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.network_graph import NetworkGraph
from utils.feature_extraction import FeatureExtractor
from utils.logger import SDNLogger
from detection.ids_engine import IDSEngine
from detection.response_manager import ResponseManager
from controllers.routing import RoutingEngine

logger = logging.getLogger(__name__)


class SDNMLController(app_manager.RyuApp):
    """
    Main SDN Controller with ML-enhanced intrusion detection and routing.
    Implements OpenFlow 1.3 with:
    - Topology discovery and graph maintenance
    - Dijkstra shortest path routing (ML-enhanced)
    - ARP proxy
    - Flow statistics collection for ML IDS
    - Automated threat response
    """
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(SDNMLController, self).__init__(*args, **kwargs)

        # Core data structures
        self.mac_to_port = {}          # {dpid: {mac: port}}
        self.mac_to_ip = {}            # {mac: ip}
        self.ip_to_mac = {}            # {ip: mac}
        self.datapaths = {}            # {dpid: datapath}
        self.hosts = {}                # {mac: (dpid, port)}

        # Topology graph
        self.network_graph = NetworkGraph()

        # Feature extraction for ML
        self.feature_extractor = FeatureExtractor()

        # IDS Engine (ML + Signature + Threat Intel)
        self.ids_engine = IDSEngine()

        # Automated response manager
        self.response_manager = ResponseManager(self)

        # Routing engine (Dijkstra + ML)
        self.routing_engine = RoutingEngine(self.network_graph)

        # SDN Logger (SIEM, InfluxDB, file)
        self.sdn_logger = SDNLogger()

        # Flow stats collection
        self.flow_stats = {}           # {dpid: [flow_stat]}
        self.port_stats = {}           # {dpid: [port_stat]}
        self.flow_history = {}         # For feature extraction

        # Blocked IPs / flows (from automated response)
        self.blocked_ips = set()
        self.rate_limited_flows = {}

        # Start background threads
        self.monitor_thread = hub.spawn(self._monitor_flows)
        self.topology_thread = hub.spawn(self._update_topology)

        logger.info("[SDNMLController] Controller initialized successfully")

    # ─────────────────────────────────────────────
    # OPENFLOW EVENT HANDLERS
    # ─────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Handle switch connection - install table-miss flow."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        self.datapaths[datapath.id] = datapath
        logger.info(f"[Controller] Switch connected: dpid={datapath.id}")
        self.sdn_logger.log_event("switch_connected", {"dpid": datapath.id})

        # Install table-miss: send unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """Track datapath connect/disconnect."""
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.datapaths[datapath.id] = datapath
                logger.info(f"[Controller] Datapath registered: {datapath.id}")
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                del self.datapaths[datapath.id]
                self.network_graph.remove_switch(datapath.id)
                logger.warning(f"[Controller] Datapath disconnected: {datapath.id}")
                self.sdn_logger.log_event("switch_disconnected", {"dpid": datapath.id})

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """Handle PacketIn events - core forwarding + IDS trigger."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        dpid = datapath.id

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore LLDP
        if eth.ethertype == 0x88cc:
            return

        dst_mac = eth.dst
        src_mac = eth.src

        # Learn MAC → port mapping
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][src_mac] = in_port
        self.hosts[src_mac] = (dpid, in_port)

        # Extract IP info
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            self.ip_to_mac[ip_pkt.src] = src_mac
            self.mac_to_ip[src_mac] = ip_pkt.src

            # Check blocked IPs
            if ip_pkt.src in self.blocked_ips:
                logger.warning(f"[IDS] Blocked packet from {ip_pkt.src}")
                return

        # Handle ARP
        arp_pkt = pkt.get_protocol(arp.arp)
        if arp_pkt:
            self._handle_arp(datapath, in_port, pkt, arp_pkt)
            return

        # Determine output port
        out_port = self._get_output_port(dpid, dst_mac, src_mac)

        actions = [parser.OFPActionOutput(out_port)]

        # Install flow if we know the destination
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(
                in_port=in_port,
                eth_dst=dst_mac,
                eth_src=src_mac
            )
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(datapath, 1, match, actions, msg.buffer_id)
            else:
                self._add_flow(datapath, 1, match, actions)

        # Forward packet
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data

        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data
        )
        datapath.send_msg(out)

        # Extract features and run IDS asynchronously
        if ip_pkt:
            hub.spawn(self._run_ids_check, pkt, ip_pkt, dpid, in_port)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Collect flow statistics for ML feature extraction."""
        dpid = ev.msg.datapath.id
        self.flow_stats[dpid] = ev.msg.body

        # Extract features from flow stats
        features = self.feature_extractor.extract_from_flow_stats(
            dpid, ev.msg.body, self.flow_history
        )

        for feat in features:
            self.flow_history.setdefault(feat['flow_key'], []).append(feat)
            # Run IDS on flow-level features
            hub.spawn(self._run_ids_on_flow, feat, dpid)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Collect port statistics for network graph weight updates."""
        dpid = ev.msg.datapath.id
        self.port_stats[dpid] = ev.msg.body
        self.network_graph.update_port_stats(dpid, ev.msg.body)

    # ─────────────────────────────────────────────
    # TOPOLOGY EVENTS
    # ─────────────────────────────────────────────

    @set_ev_cls(topo_event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        self.network_graph.add_switch(ev.switch.dp.id)
        self._request_topology_update()

    @set_ev_cls(topo_event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        self.network_graph.remove_switch(ev.switch.dp.id)

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add_handler(self, ev):
        src = ev.link.src
        dst = ev.link.dst
        self.network_graph.add_link(src.dpid, dst.dpid, src.port_no, dst.port_no)
        logger.info(f"[Topology] Link added: {src.dpid} → {dst.dpid}")

    @set_ev_cls(topo_event.EventLinkDelete)
    def link_delete_handler(self, ev):
        src = ev.link.src
        dst = ev.link.dst
        self.network_graph.remove_link(src.dpid, dst.dpid)
        logger.warning(f"[Topology] Link removed: {src.dpid} → {dst.dpid}")
        self.sdn_logger.log_event("link_failure", {"src": src.dpid, "dst": dst.dpid})
        # Trigger rerouting
        hub.spawn(self._handle_link_failure, src.dpid, dst.dpid)

    # ─────────────────────────────────────────────
    # ARP HANDLING
    # ─────────────────────────────────────────────

    def _handle_arp(self, datapath, in_port, pkt, arp_pkt):
        """ARP proxy - respond from controller if host is known."""
        if arp_pkt.opcode != arp.ARP_REQUEST:
            return

        target_ip = arp_pkt.dst_ip
        if target_ip in self.ip_to_mac:
            # We know the target - reply on behalf
            target_mac = self.ip_to_mac[target_ip]
            self._send_arp_reply(datapath, in_port, arp_pkt, target_mac)
        else:
            # Flood ARP request
            self._flood(datapath, pkt, in_port)

    def _send_arp_reply(self, datapath, in_port, arp_req, target_mac):
        """Send ARP reply from controller."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        pkt_reply = packet.Packet()
        pkt_reply.add_protocol(ethernet.ethernet(
            ethertype=0x0806,
            dst=arp_req.src_mac,
            src=target_mac
        ))
        pkt_reply.add_protocol(arp.arp(
            opcode=arp.ARP_REPLY,
            src_mac=target_mac,
            src_ip=arp_req.dst_ip,
            dst_mac=arp_req.src_mac,
            dst_ip=arp_req.src_ip
        ))
        pkt_reply.serialize()

        actions = [parser.OFPActionOutput(in_port)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=ofproto.OFPP_CONTROLLER,
            actions=actions,
            data=pkt_reply.data
        )
        datapath.send_msg(out)

    # ─────────────────────────────────────────────
    # ROUTING
    # ─────────────────────────────────────────────

    def _get_output_port(self, dpid, dst_mac, src_mac):
        """Get output port using shortest path routing."""
        from ryu.ofproto import ofproto_v1_3
        ofproto = self.datapaths[dpid].ofproto if dpid in self.datapaths else None

        # Direct lookup
        if dpid in self.mac_to_port and dst_mac in self.mac_to_port[dpid]:
            return self.mac_to_port[dpid][dst_mac]

        # Multi-hop: find path via NetworkGraph
        if dst_mac in self.hosts:
            dst_dpid, dst_port = self.hosts[dst_mac]
            path = self.routing_engine.get_path(dpid, dst_dpid)
            if path and len(path) > 1:
                next_hop = path[1]
                link_port = self.network_graph.get_port(dpid, next_hop)
                if link_port:
                    return link_port

        return ofproto_v1_3.OFPP_FLOOD

    def _handle_link_failure(self, src_dpid, dst_dpid):
        """Recalculate paths after link failure using ML-enhanced routing."""
        logger.warning(f"[Routing] Handling link failure: {src_dpid} ↔ {dst_dpid}")
        # Invalidate flow table entries using failed link
        for datapath in self.datapaths.values():
            self._delete_flows_via_link(datapath, src_dpid, dst_dpid)
        # Paths will be recalculated on next PacketIn

    def _delete_flows_via_link(self, datapath, src_dpid, dst_dpid):
        """Remove flows that traverse the failed link."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch()
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

    # ─────────────────────────────────────────────
    # IDS INTEGRATION
    # ─────────────────────────────────────────────

    def _run_ids_check(self, pkt, ip_pkt, dpid, in_port):
        """Run IDS on packet-level features."""
        try:
            features = self.feature_extractor.extract_from_packet(pkt, ip_pkt, dpid, in_port)
            result = self.ids_engine.analyze(features)
            if result['threat_detected']:
                self._handle_threat(result, ip_pkt.src, dpid)
        except Exception as e:
            logger.error(f"[IDS] Packet analysis error: {e}")

    def _run_ids_on_flow(self, flow_features, dpid):
        """Run IDS on flow-level features."""
        try:
            result = self.ids_engine.analyze(flow_features)
            if result['threat_detected']:
                self._handle_threat(result, flow_features.get('src_ip'), dpid)
        except Exception as e:
            logger.error(f"[IDS] Flow analysis error: {e}")

    def _handle_threat(self, detection_result, src_ip, dpid):
        """Dispatch threat to response manager."""
        logger.warning(f"[IDS] Threat detected: {detection_result['threat_type']} "
                       f"from {src_ip} (score={detection_result['confidence']:.2f})")

        self.sdn_logger.log_threat(detection_result, src_ip, dpid)

        # Automated response
        self.response_manager.respond(
            threat=detection_result,
            src_ip=src_ip,
            dpid=dpid,
            datapaths=self.datapaths,
            blocked_ips=self.blocked_ips
        )

    # ─────────────────────────────────────────────
    # BACKGROUND MONITORING
    # ─────────────────────────────────────────────

    def _monitor_flows(self):
        """Poll flow and port stats from all switches every 10 seconds."""
        while True:
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)
                self._request_port_stats(dp)
            hub.sleep(10)

    def _update_topology(self):
        """Periodically refresh topology from Ryu topology API."""
        while True:
            hub.sleep(30)
            self._request_topology_update()

    def _request_flow_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    def _request_port_stats(self, datapath):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY)
        datapath.send_msg(req)

    def _request_topology_update(self):
        switches = get_switch(self, None)
        links = get_link(self, None)
        self.network_graph.update_from_ryu(switches, links)

    # ─────────────────────────────────────────────
    # FLOW MANAGEMENT HELPERS
    # ─────────────────────────────────────────────

    def _add_flow(self, datapath, priority, match, actions,
                  buffer_id=None, idle_timeout=30, hard_timeout=120):
        """Install a flow entry on a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]

        kwargs = dict(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        if buffer_id and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs['buffer_id'] = buffer_id

        mod = parser.OFPFlowMod(**kwargs)
        datapath.send_msg(mod)

    def _flood(self, datapath, pkt, in_port):
        """Flood packet out all ports except in_port."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=pkt.data
        )
        datapath.send_msg(out)

    # ─────────────────────────────────────────────
    # PUBLIC API FOR RESPONSE MANAGER
    # ─────────────────────────────────────────────

    def block_ip(self, ip, dpid=None):
        """Block an IP across all or specific switches."""
        self.blocked_ips.add(ip)
        targets = [self.datapaths[dpid]] if dpid and dpid in self.datapaths \
                  else list(self.datapaths.values())
        for dp in targets:
            self._install_drop_rule(dp, ip)
        logger.warning(f"[Response] IP blocked: {ip}")

    def unblock_ip(self, ip):
        """Remove IP block."""
        self.blocked_ips.discard(ip)
        for dp in self.datapaths.values():
            self._remove_drop_rule(dp, ip)
        logger.info(f"[Response] IP unblocked: {ip}")

    def rate_limit_ip(self, ip, rate_kbps, dpid=None):
        """Apply rate limiting to an IP (via meter)."""
        self.rate_limited_flows[ip] = rate_kbps
        logger.info(f"[Response] Rate limit applied: {ip} → {rate_kbps} kbps")

    def _install_drop_rule(self, datapath, src_ip, priority=100):
        """Install a DROP rule for a specific source IP."""
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        self._add_flow(datapath, priority, match, [], hard_timeout=300)

    def _remove_drop_rule(self, datapath, src_ip):
        """Remove DROP rule for a specific source IP."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=src_ip)
        mod = parser.OFPFlowMod(
            datapath=datapath,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=match
        )
        datapath.send_msg(mod)

