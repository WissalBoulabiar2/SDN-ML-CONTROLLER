"""
dhcp_controller.py — DHCP Snooping + Mitigation (module isolé)
===============================================================
Fixes appliqués :
  4. Auto-block immédiat sur rogue DHCP server (plus de pending confirm)
  3. Modularisé : SharedState injecté, ne dépend plus de stp_core

REST API :
  GET  /dhcp-snooping/status
  GET  /stp/dhcp-snooping/alerts
  POST /stp/dhcp-snooping/trust/<dpid>/<port>
  DELETE /stp/dhcp-snooping/trust/<dpid>/<port>
  POST /stp/dhcp-snooping/unblock/<dpid>/<port>
  GET  /dhcp-mitigation/status
  GET  /dhcp-mitigation/threats
  POST /dhcp-mitigation/quarantine/<dpid>/<port>
  POST /dhcp-mitigation/release/<dpid>/<port>
  POST /dhcp-mitigation/reset-score/<dpid>/<port>
"""

import logging
import json
import time
import threading
from typing import Optional

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, udp
from ryu.lib import stplib
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

from controllers.shared_state import SharedState
from controllers.dhcp_mitigation import DHCPMitigationLayer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes DHCP ──────────────────────────────────────────────────────────
DHCP_SERVER_PORT = 67
DHCP_CLIENT_PORT = 68
DHCP_MAGIC_COOKIE = b'\x63\x82\x53\x63'

DHCP_DISCOVER = 1
DHCP_OFFER    = 2
DHCP_REQUEST  = 3
DHCP_ACK      = 5
DHCP_NAK      = 6

DHCP_SERVER_MSG_TYPES = {DHCP_OFFER, DHCP_ACK, DHCP_NAK}

BPDU_DST_STD  = '01:80:c2:00:00:00'
BPDU_DST_PVST = '01:00:0c:cc:cc:cd'


# ── Détecteur DHCP ───────────────────────────────────────────────────────────

class DHCPDetector:
    @staticmethod
    def detect(raw: bytes) -> Optional[dict]:
        try:
            pkt  = packet.Packet(raw)
            eth  = pkt.get_protocol(ethernet.ethernet)
            ip4  = pkt.get_protocol(ipv4.ipv4)
            udp_ = pkt.get_protocol(udp.udp)

            if not (eth and ip4 and udp_):
                return None
            if udp_.src_port not in (DHCP_SERVER_PORT, DHCP_CLIENT_PORT):
                return None
            if udp_.dst_port not in (DHCP_SERVER_PORT, DHCP_CLIENT_PORT):
                return None

            payload = DHCPDetector._udp_payload(raw)
            if payload is None or len(payload) < 240:
                return None
            if payload[236:240] != DHCP_MAGIC_COOKIE:
                return None

            msg_type = DHCPDetector._option53(payload[240:])
            if msg_type is None:
                return None

            client_mac = ':'.join(f'{b:02x}' for b in payload[28:34])
            offered_ip = '.'.join(str(b) for b in payload[16:20])
            if offered_ip == '0.0.0.0':
                offered_ip = None

            return {
                'msg_type':      msg_type,
                'client_mac':    client_mac,
                'offered_ip':    offered_ip,
                'src_ip':        ip4.src,
                'src_mac':       eth.src,
                'is_server_msg': msg_type in DHCP_SERVER_MSG_TYPES,
            }
        except Exception as e:
            logger.debug(f"[DHCP] parse error: {e}")
            return None

    @staticmethod
    def _udp_payload(raw: bytes) -> Optional[bytes]:
        try:
            pkt  = packet.Packet(raw)
            udp_ = pkt.get_protocol(udp.udp)
            if not udp_:
                return None
            return udp_.data if isinstance(udp_.data, (bytes, bytearray)) else None
        except Exception:
            return None

    @staticmethod
    def _option53(opts: bytes) -> Optional[int]:
        i = 0
        while i < len(opts):
            code = opts[i]
            if code == 255:
                break
            if code == 0:
                i += 1
                continue
            if i + 1 >= len(opts):
                break
            length = opts[i + 1]
            if code == 53 and length >= 1:
                return opts[i + 2]
            i += 2 + length
        return None


# ── DHCP Snooping Manager ────────────────────────────────────────────────────

class DHCPSnoopingManager:
    def __init__(self):
        self._lock   = threading.Lock()
        self.trusted = set()
        self.blocked = set()
        self.binding = {}
        self.alerts  = []

    def set_trusted(self, dpid, port):
        with self._lock:
            self.trusted.add((dpid, port))
        logger.info(f"[DHCP Snooping] Trusted port: dpid={dpid} port={port}")

    def unset_trusted(self, dpid, port):
        with self._lock:
            self.trusted.discard((dpid, port))
        logger.info(f"[DHCP Snooping] Untrusted port: dpid={dpid} port={port}")

    def is_trusted(self, dpid, port) -> bool:
        with self._lock:
            return (dpid, port) in self.trusted

    def record_binding(self, client_mac, offered_ip, dpid, port):
        with self._lock:
            self.binding[client_mac] = {
                'ip':        offered_ip,
                'dpid':      dpid,
                'port':      port,
                'leased_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        logger.info(f"[DHCP Snooping] Binding: {client_mac} → {offered_ip} dpid={dpid} port={port}")

    def mark_blocked(self, dpid, port):
        with self._lock:
            self.blocked.add((dpid, port))

    def unmark_blocked(self, dpid, port):
        with self._lock:
            self.blocked.discard((dpid, port))

    def is_blocked(self, dpid, port) -> bool:
        with self._lock:
            return (dpid, port) in self.blocked

    def add_alert(self, dpid, port, details):
        with self._lock:
            alert = {
                'type':      'ROGUE_DHCP_SERVER',
                'severity':  'CRITICAL',
                'dpid':      dpid,
                'port':      port,
                'details':   details,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            self.alerts.append(alert)
            self.alerts = self.alerts[-100:]
            return alert

    def get_alerts(self):
        with self._lock:
            return list(reversed(self.alerts))

    def get_status(self):
        with self._lock:
            return {
                'trusted_ports':       [{'dpid': d, 'port': p} for (d, p) in self.trusted],
                'blocked_rogue_ports': [{'dpid': d, 'port': p} for (d, p) in self.blocked],
                'binding_table':       dict(self.binding),
                'rogue_alerts':        len(self.alerts),
            }


# ── REST API DHCP ─────────────────────────────────────────────────────────────

DHCP_APP_KEY = 'dhcp_app'


class DHCPRestAPI(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[DHCP_APP_KEY]

    def _ok(self, data):
        return Response(
            status=200, content_type='application/json', charset='utf-8',
            body=json.dumps(data, indent=2, default=str).encode('utf-8')
        )

    def _err(self, msg, status=404):
        return Response(
            status=status, content_type='application/json', charset='utf-8',
            body=json.dumps({'error': msg}).encode('utf-8')
        )

    # ── DHCP Snooping ─────────────────────────────────────────────────────────

    @route('dhcp', '/stp/dhcp-snooping/trust/{dpid}/{port_no}', methods=['POST'])
    def set_dhcp_trust(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_snooping.set_trusted(dpid, port_no)
        return self._ok({
            'result': f'DHCP Trusted port: dpid={dpid} port={port_no}',
            'effect': 'DHCPOFFER/ACK/NAK autorise sur ce port uniquement',
        })

    @route('dhcp', '/stp/dhcp-snooping/trust/{dpid}/{port_no}', methods=['DELETE'])
    def unset_dhcp_trust(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_snooping.unset_trusted(dpid, port_no)
        return self._ok({'result': f'Port dpid={dpid} port={port_no} → UNTRUSTED'})

    @route('dhcp', '/dhcp-snooping/status', methods=['GET'])
    def get_dhcp_status(self, req, **kwargs):
        return self._ok(self.app.dhcp_snooping.get_status())

    @route('dhcp', '/stp/dhcp-snooping/alerts', methods=['GET'])
    def get_dhcp_alerts(self, req, **kwargs):
        alerts = self.app.dhcp_snooping.get_alerts()
        return self._ok({'count': len(alerts), 'alerts': alerts})

    @route('dhcp', '/stp/dhcp-snooping/unblock/{dpid}/{port_no}', methods=['POST'])
    def dhcp_unblock(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        if dpid not in self.app.datapaths:
            return self._err('Switch non connecte')
        self.app._do_unblock(self.app.datapaths[dpid], port_no)
        self.app.dhcp_snooping.unmark_blocked(dpid, port_no)
        return self._ok({'result': f'Port DEBLOQUE (DHCP): dpid={dpid} port={port_no}'})

    # ── DHCP Mitigation ───────────────────────────────────────────────────────

    @route('dhcp', '/dhcp-mitigation/status', methods=['GET'])
    def get_mitigation_status(self, req, **kwargs):
        return self._ok(self.app.dhcp_mitigation.get_full_status())

    @route('dhcp', '/dhcp-mitigation/threats', methods=['GET'])
    def get_threat_history(self, req, **kwargs):
        return self._ok({
            'scores':  self.app.dhcp_mitigation.threat_scorer.get_all_scores(),
            'history': self.app.dhcp_mitigation.threat_scorer.get_history(50),
        })

    @route('dhcp', '/dhcp-mitigation/quarantine/{dpid}/{port_no}', methods=['POST'])
    def force_quarantine(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_mitigation.quarantine_mgr.quarantine(dpid, port_no)
        return self._ok({'result': f'Quarantine forcee: dpid={dpid} port={port_no}'})

    @route('dhcp', '/dhcp-mitigation/release/{dpid}/{port_no}', methods=['POST'])
    def release_quarantine(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        if dpid not in self.app.datapaths:
            return self._err('Switch non connecte')
        ok, msg = self.app.dhcp_mitigation.try_release(
            self.app.datapaths[dpid], port_no)
        if not ok:
            return self._err(msg, status=403)
        self.app._do_unblock(self.app.datapaths[dpid], port_no)
        return self._ok({'result': msg})

    @route('dhcp', '/dhcp-mitigation/reset-score/{dpid}/{port_no}', methods=['POST'])
    def reset_threat_score(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_mitigation.reset_threat_score(dpid, port_no)
        return self._ok({'result': f'Score reinitialise: dpid={dpid} port={port_no}'})


# ── Contrôleur DHCP ──────────────────────────────────────────────────────────

class DHCPController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {
        'wsgi':         WSGIApplication,
        'shared_state': SharedState,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        shared           = kwargs['shared_state']
        self.datapaths   = shared.datapaths
        self.mac_to_port = shared.mac_to_port

        self.dhcp_snooping  = DHCPSnoopingManager()
        self.dhcp_mitigation = DHCPMitigationLayer(self)

        wsgi = kwargs['wsgi']
        wsgi.register(DHCPRestAPI, {DHCP_APP_KEY: self})

        logger.info("=" * 60)
        logger.info("  [DHCP] DHCP Snooping + Mitigation Layer")
        logger.info("  Rogue DHCP → blocage immédiat automatique (FIX 4)")
        logger.info("")
        logger.info("  GET  /dhcp-snooping/status")
        logger.info("  GET  /stp/dhcp-snooping/alerts")
        logger.info("  POST /stp/dhcp-snooping/trust/<dpid>/<port>")
        logger.info("  DELETE /stp/dhcp-snooping/trust/<dpid>/<port>")
        logger.info("  POST /stp/dhcp-snooping/unblock/<dpid>/<port>")
        logger.info("  GET  /dhcp-mitigation/status")
        logger.info("  GET  /dhcp-mitigation/threats")
        logger.info("  POST /dhcp-mitigation/quarantine/<dpid>/<port>")
        logger.info("  POST /dhcp-mitigation/release/<dpid>/<port>")
        logger.info("  POST /dhcp-mitigation/reset-score/<dpid>/<port>")
        logger.info("=" * 60)

    # ── Packet in ─────────────────────────────────────────────────────────────

    @set_ev_cls(stplib.EventPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg      = ev.msg
        datapath = msg.datapath
        in_port  = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # Ignore les BPDUs (traités par stp_core)
        if eth.dst.lower() in (BPDU_DST_STD, BPDU_DST_PVST):
            return

        # ── Mitigation layer (rate-limiting, scoring) ─────────────────────────
        dhcp_pre = DHCPDetector.detect(msg.data)
        ip4_pkt  = pkt.get_protocol(ipv4.ipv4)
        src_ip   = ip4_pkt.src if ip4_pkt else ''
        if not self.dhcp_mitigation.inspect(datapath, in_port, dhcp_pre, src_ip, eth.src):
            return

        # ── Snooping : bloque les rogue DHCP servers (FIX 4 : auto-block) ────
        dhcp_info = DHCPDetector.detect(msg.data)
        if dhcp_info is not None:
            self._check_dhcp_snooping(datapath, in_port, dhcp_info)

    # ── DHCP Snooping ─────────────────────────────────────────────────────────

    def _check_dhcp_snooping(self, datapath, in_port, dhcp_info: dict) -> bool:
        """
        FIX 4 : blocage immédiat sans confirmation REST si rogue DHCP détecté.
        Retourne True si le paquet est légitime, False s'il est droppé.
        """
        dpid     = datapath.id
        msg_type = dhcp_info['msg_type']

        if msg_type not in DHCP_SERVER_MSG_TYPES:
            return True

        if self.dhcp_snooping.is_trusted(dpid, in_port):
            if msg_type == DHCP_ACK:
                self.dhcp_snooping.record_binding(
                    dhcp_info['client_mac'], dhcp_info['offered_ip'], dpid, in_port)
            return True

        # Port non trusted + message serveur → rogue DHCP → blocage immédiat
        details = (
            f"Rogue server: IP={dhcp_info['src_ip']} MAC={dhcp_info['src_mac']} "
            f"MsgType={msg_type} client={dhcp_info['client_mac']}"
        )
        logger.critical(
            f"!!! [DHCP SNOOPING] {details} dpid={dpid} port={in_port} "
            f"→ AUTO-BLOCK immediat"
        )

        self.dhcp_snooping.add_alert(dpid, in_port, details)
        self.dhcp_snooping.mark_blocked(dpid, in_port)

        # FIX 4 : blocage immédiat, sans pending.add()
        self._do_block(datapath, in_port, "DHCP-Snooping")

        from controllers.stp_core import PORT_ICONS  # évite import circulaire au niveau module
        self._update_port_state(dpid, in_port, 'dhcp_rogue_err', 'DHCP',
                                'Rogue DHCP server detected - auto-blocked')
        return False

    def _update_port_state(self, dpid, port, state, protocol, note=''):
        """
        Met à jour l'état d'un port dans le registre partagé.
        Évite de dupliquer port_states dans stp_core : on notifie via log.
        """
        logger.info(
            f"[DHCP] Port state update: dpid={dpid} port={port} "
            f"state={state} [{protocol}] {note}"
        )

    # ── FlowMod helpers (dupliqués depuis stp_core pour isolation totale) ─────

    def _do_block(self, datapath, port, protocol):
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto

        for bpdu_dst in ('01:80:c2:00:00:00', '01:00:0c:cc:cc:cd'):
            match_bpdu = parser.OFPMatch(in_port=port, eth_dst=bpdu_dst)
            mod_bpdu   = parser.OFPFlowMod(
                datapath=datapath, priority=200,
                command=ofproto.OFPFC_ADD,
                match=match_bpdu,
                instructions=[parser.OFPInstructionActions(
                    ofproto.OFPIT_APPLY_ACTIONS,
                    [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                           ofproto.OFPCML_NO_BUFFER)]
                )]
            )
            datapath.send_msg(mod_bpdu)

        match_data = parser.OFPMatch(in_port=port)
        mod_data   = parser.OFPFlowMod(
            datapath=datapath, priority=100,
            command=ofproto.OFPFC_ADD,
            match=match_data, instructions=[]
        )
        datapath.send_msg(mod_data)

        logger.info(
            f"[ACTION] Port BLOQUE: dpid={datapath.id} port={port} [{protocol}] "
            f"(data=DROP | BPDUs=passthrough)"
        )

    def _do_unblock(self, datapath, port):
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto

        for bpdu_dst in ('01:80:c2:00:00:00', '01:00:0c:cc:cc:cd'):
            match_bpdu = parser.OFPMatch(in_port=port, eth_dst=bpdu_dst)
            mod_del    = parser.OFPFlowMod(
                datapath=datapath, priority=200,
                command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                match=match_bpdu
            )
            datapath.send_msg(mod_del)

        match_data = parser.OFPMatch(in_port=port)
        mod_data   = parser.OFPFlowMod(
            datapath=datapath, priority=100,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            match=match_data
        )
        datapath.send_msg(mod_data)

        logger.info(f"[ACTION] Port DEBLOQUE: dpid={datapath.id} port={port}")
