"""
controller_stp.py — STP Multi-Protocole SEMI-MANUEL + Sécurité + DHCP Snooping
================================================================================
Fonctionnalités:
  1. stplib détecte les boucles → blocage automatique des ports NON_DESIGNATED
  2. RSTP / MSTP / PVST / PVST+ détectés via analyse BPDU raw
  3. BPDU Guard  — bloque immédiatement les ports edge qui reçoivent des BPDUs
  4. Root Guard  — empêche un port de devenir root port (Root Bridge Hijacking)
  5. DHCP Snooping — seuls les ports "trusted" peuvent envoyer des réponses DHCP
  6. Edge port detection — ports PC/NAT jamais bloqués (détection LLDP dynamique)

Attaques contrées:
  • Root Bridge Hijacking → Root Guard
  • Rogue DHCP Server     → DHCP Snooping + Mitigation Layer
  • BPDU sur port host    → BPDU Guard

REST API:
  GET  /stp/status
  GET  /stp/pending
  GET  /stp/security/alerts
  POST /stp/confirm/<id>
  POST /stp/cancel/<id>
  POST /stp/unblock/<dpid>/<port>
  POST /stp/security/bpdu-guard/<dpid>/<port>
  POST /stp/security/root-guard/<dpid>/<port>
  DELETE /stp/security/guard/<dpid>/<port>

  ── DHCP Snooping ──────────────────────────────────────────────
  POST /stp/dhcp-snooping/trust/<dpid>/<port>
  DELETE /stp/dhcp-snooping/trust/<dpid>/<port>
  GET  /dhcp-snooping/status
  POST /stp/dhcp-snooping/unblock/<dpid>/<port>

  ── DHCP Mitigation ────────────────────────────────────────────
  GET  /dhcp-mitigation/status
  GET  /dhcp-mitigation/threats
  POST /dhcp-mitigation/quarantine/<dpid>/<port>
  POST /dhcp-mitigation/release/<dpid>/<port>
  POST /dhcp-mitigation/reset-score/<dpid>/<port>

Usage:
  PYTHONPATH=. ryu-manager controllers/controller_stp.py --observe-links
"""

from controllers.sdn_rl_routing import QRoutingAgent, SDNNetwork
from controllers.dhcp_mitigation import DHCPMitigationLayer
import struct
import logging
import json
import time
import threading
from collections import defaultdict
from typing import Dict, Optional, Union, List, Any

import networkx as nx

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, ipv4, udp
from ryu.lib import stplib
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Constantes BPDU ──────────────────────────────────────────────────────────
BPDU_DST_STD  = '01:80:c2:00:00:00'
BPDU_DST_PVST = '01:00:0c:cc:cc:cd'

PROTO_VER_STP  = 0x00
PROTO_VER_RSTP = 0x02
PROTO_VER_MSTP = 0x03

FLAG_PROPOSAL   = 0x02
FLAG_LEARNING   = 0x10
FLAG_FORWARDING = 0x20

MAX_PRIORITY = 65535

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

PORT_ICONS = {
    'blocking':          '🔴 BLOCKING',
    'listening':         '🟡 LISTENING',
    'learning':          '🟠 LEARNING',
    'forwarding':        '🟢 FORWARDING',
    'discarding':        '🔴 DISCARDING',
    'disabled':          '⚫ DISABLED',
    'root_inconsistent': '🚨 ROOT-INCONSISTENT',
    'bpdu_guard_err':    '🚫 BPDU-GUARD-ERR-DISABLE',
    'dhcp_rogue_err':    '🚫 DHCP-ROGUE-ERR-DISABLE',
}

STPLIB_STATE_MAP = {
    stplib.PORT_STATE_DISABLE: 'disabled',
    stplib.PORT_STATE_BLOCK:   'blocking',
    stplib.PORT_STATE_LISTEN:  'listening',
    stplib.PORT_STATE_LEARN:   'learning',
    stplib.PORT_STATE_FORWARD: 'forwarding',
}


# ── Détecteur BPDU ───────────────────────────────────────────────────────────

class BPDUDetector:
    @staticmethod
    def detect(eth_dst: str, raw: bytes) -> dict:
        info = {
            'protocol': 'UNKNOWN', 'version': 0, 'flags': 0,
            'vlan': None, 'instance': None,
            'root_id': 'N/A', 'bridge_id': 'N/A',
            'root_priority': MAX_PRIORITY, 'root_mac': 'N/A',
        }
        try:
            offset = 14
            if len(raw) < offset + 4:
                return info
            if raw[offset] == 0x42 and raw[offset+1] == 0x42:
                offset += 3
            elif raw[offset:offset+3] == b'\xaa\xaa\x03':
                offset += 8
            if len(raw) < offset + 4:
                return info

            version   = raw[offset+2]
            bpdu_type = raw[offset+3]
            info['version'] = version
            if len(raw) > offset+4:
                info['flags'] = raw[offset+4]

            dst = eth_dst.lower()
            if dst == BPDU_DST_PVST:
                info['vlan']     = BPDUDetector._vlan(raw)
                info['protocol'] = 'PVST+' if version >= PROTO_VER_RSTP else 'PVST'
            elif version == PROTO_VER_MSTP:
                info['protocol'] = 'MSTP'
                if len(raw) > offset+102:
                    info['instance'] = raw[offset+102] & 0x0F
            elif version == PROTO_VER_RSTP:
                info['protocol'] = 'RSTP'
            elif version == PROTO_VER_STP:
                info['protocol'] = 'STP'

            if bpdu_type == 0x00 and len(raw) >= offset+35:
                root_raw = raw[offset+5:offset+13]
                info['root_id']       = BPDUDetector._bid(root_raw)
                info['bridge_id']     = BPDUDetector._bid(raw[offset+17:offset+25])
                if len(root_raw) >= 8:
                    info['root_priority'] = struct.unpack('!H', root_raw[:2])[0]
                    info['root_mac']      = ':'.join(f'{b:02x}' for b in root_raw[2:8])
        except Exception as e:
            logger.debug(f"[BPDU] parse error: {e}")
        return info

    @staticmethod
    def _bid(raw):
        if len(raw) < 8: return 'N/A'
        return f"{struct.unpack('!H', raw[:2])[0]}/{':'.join(f'{b:02x}' for b in raw[2:8])}"

    @staticmethod
    def _vlan(raw):
        try:
            if raw[12:14] == b'\x81\x00':
                return struct.unpack('!H', raw[14:16])[0] & 0x0FFF
        except Exception:
            pass
        return None


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
            length = opts[i+1]
            if code == 53 and length >= 1:
                return opts[i+2]
            i += 2 + length
        return None


# ── DHCP Snooping Manager ────────────────────────────────────────────────────

class DHCPSnoopingManager:
    def __init__(self):
        self._lock        = threading.Lock()
        self.trusted      = set()
        self.blocked      = set()
        self.binding      = {}
        self.alerts       = []

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
        logger.info(f"[DHCP Snooping] Binding: {client_mac} → {offered_ip}  "
                    f"dpid={dpid} port={port}")

    def get_binding(self, client_mac):
        with self._lock:
            return self.binding.get(client_mac)

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
                'trusted_ports':      [{'dpid': d, 'port': p} for (d, p) in self.trusted],
                'blocked_rogue_ports':[{'dpid': d, 'port': p} for (d, p) in self.blocked],
                'binding_table':      dict(self.binding),
                'rogue_alerts':       len(self.alerts),
            }


# ── Gestionnaire d'actions en attente ─────────────────────────────────────────

class PendingActions:
    def __init__(self, timeout=120):
        self._lock    = threading.Lock()
        self._actions = {}
        self._counter = 0
        self.timeout  = timeout

    def add(self, dpid, port, protocol, reason, details=None) -> int:
        with self._lock:
            self._counter += 1
            aid = self._counter
            self._actions[aid] = {
                'id': aid, 'dpid': dpid, 'port_no': port,
                'protocol': protocol, 'reason': reason,
                'details': details or {}, 'created_at': time.time(),
            }
            return aid

    def confirm(self, aid):
        with self._lock: return self._actions.pop(aid, None)

    def cancel(self, aid):
        with self._lock: return self._actions.pop(aid, None)

    def all(self):
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._actions.items()
                      if now - v['created_at'] > self.timeout]:
                del self._actions[k]
            return [{**v, 'remaining_s': int(self.timeout - (now - v['created_at']))}
                    for v in self._actions.values()]

    def already_pending(self, dpid, port):
        with self._lock:
            return any(v['dpid'] == dpid and v['port_no'] == port
                       for v in self._actions.values())


# ── Gestionnaire de sécurité STP ─────────────────────────────────────────────

class STPSecurityManager:
    def __init__(self):
        self._lock      = threading.Lock()
        self.protected  = {}
        self.known_root = None
        self.alerts     = []

    def enable_bpdu_guard(self, dpid, port):
        with self._lock:
            k = (dpid, port)
            self.protected[k] = 'both' if self.protected.get(k) == 'root_guard' else 'bpdu_guard'
        logger.info(f"[SECURITY] BPDU Guard ACTIVE: dpid={dpid} port={port}")

    def enable_root_guard(self, dpid, port):
        with self._lock:
            k = (dpid, port)
            self.protected[k] = 'both' if self.protected.get(k) == 'bpdu_guard' else 'root_guard'
        logger.info(f"[SECURITY] Root Guard ACTIVE: dpid={dpid} port={port}")

    def disable_guard(self, dpid, port):
        with self._lock:
            self.protected.pop((dpid, port), None)

    def has_bpdu_guard(self, dpid, port):
        with self._lock:
            return self.protected.get((dpid, port)) in ('bpdu_guard', 'both')

    def has_root_guard(self, dpid, port):
        with self._lock:
            return self.protected.get((dpid, port)) in ('root_guard', 'both')

    def update_root(self, priority, mac, dpid):
        with self._lock:
            self.known_root = {
                'priority': priority, 'mac': mac,
                'dpid': dpid, 'seen_at': time.strftime('%H:%M:%S'),
            }

    def is_superior_bpdu(self, priority, mac) -> bool:
        with self._lock:
            if self.known_root is None:
                return False
            kp, km = self.known_root['priority'], self.known_root['mac']
            return priority < kp or (priority == kp and mac < km)

    def add_alert(self, alert_type, dpid, port, details):
        with self._lock:
            alert = {
                'type':      alert_type,
                'severity':  'CRITICAL' if 'HIJACK' in alert_type else 'WARNING',
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
                'protected_ports': {
                    f"dpid={d}/port={p}": v
                    for (d, p), v in self.protected.items()
                },
                'known_root':  self.known_root,
                'alert_count': len(self.alerts),
            }


# ── REST API ──────────────────────────────────────────────────────────────────

STP_APP_KEY = 'stp_app'

class STPRestAPI(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[STP_APP_KEY]

    def _ok(self, data):
        return Response(status=200, content_type='application/json', charset='utf-8',
                        body=json.dumps(data, indent=2, default=str).encode('utf-8'))

    def _err(self, msg, status=404):
        return Response(status=status, content_type='application/json', charset='utf-8',
                        body=json.dumps({'error': msg}).encode('utf-8'))

    # ── STP ──────────────────────────────────────────────────────────────────

    @route('stp', '/stp/status', methods=['GET'])
    def get_status(self, req, **kwargs):
        return self._ok({
            'protocols_detected': dict(self.app.proto_count),
            'inter_switch_ports': [
                {'dpid': d, 'port': p}
                for (d, p) in sorted(self.app._inter_switch_ports)
            ],
            'port_states': {f"dpid={d}/port={p}": v
                            for (d, p), v in self.app.port_states.items()},
            'blocked_ports': [{'dpid': d, 'port': p}
                              for (d, p), v in self.app.port_states.items()
                              if v.get('state') in ('blocking', 'discarding',
                                                    'root_inconsistent',
                                                    'bpdu_guard_err',
                                                    'dhcp_rogue_err')],
            'stp_security':  self.app.security.get_status(),
            'dhcp_snooping': self.app.dhcp_snooping.get_status(),
        })

    @route('stp', '/stp/pending', methods=['GET'])
    def get_pending(self, req, **kwargs):
        p = self.app.pending.all()
        return self._ok({'count': len(p), 'actions': p})

    @route('stp', '/stp/confirm/{action_id}', methods=['POST'])
    def confirm(self, req, action_id, **kwargs):
        action = self.app.pending.confirm(int(action_id))
        if not action:
            return self._err('Action introuvable ou expiree')
        dpid, port = action['dpid'], action['port_no']
        if dpid not in self.app.datapaths:
            return self._err(f'Switch dpid={dpid} non connecte')
        self.app._do_block(self.app.datapaths[dpid], port, action['protocol'])
        return self._ok({'result': f'Port BLOQUE: dpid={dpid} port={port}'})

    @route('stp', '/stp/cancel/{action_id}', methods=['POST'])
    def cancel(self, req, action_id, **kwargs):
        action = self.app.pending.cancel(int(action_id))
        if not action:
            return self._err('Action introuvable')
        return self._ok({'result': f'Action #{action_id} annulee'})

    @route('stp', '/stp/unblock/{dpid}/{port_no}', methods=['POST'])
    def unblock(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        if dpid not in self.app.datapaths:
            return self._err('Switch non connecte')
        self.app._do_unblock(self.app.datapaths[dpid], port_no)
        return self._ok({'result': f'Port DEBLOQUE: dpid={dpid} port={port_no}'})

    # ── STP Security ─────────────────────────────────────────────────────────

    @route('stp', '/stp/security/alerts', methods=['GET'])
    def get_alerts(self, req, **kwargs):
        alerts = self.app.security.get_alerts()
        return self._ok({'count': len(alerts), 'alerts': alerts})

    @route('stp', '/stp/security/bpdu-guard/{dpid}/{port_no}', methods=['POST'])
    def set_bpdu_guard(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.security.enable_bpdu_guard(dpid, port_no)
        return self._ok({
            'result': f'BPDU Guard ACTIVE: dpid={dpid} port={port_no}',
            'effect': 'Tout BPDU recu sur ce port → blocage immediat sans confirmation',
        })

    @route('stp', '/stp/security/root-guard/{dpid}/{port_no}', methods=['POST'])
    def set_root_guard(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.security.enable_root_guard(dpid, port_no)
        return self._ok({
            'result': f'Root Guard ACTIVE: dpid={dpid} port={port_no}',
            'effect': 'BPDU superieur detecte → root-inconsistent + alerte CRITICAL',
        })

    @route('stp', '/stp/security/guard/{dpid}/{port_no}', methods=['DELETE'])
    def del_guard(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.security.disable_guard(dpid, port_no)
        return self._ok({'result': f'Guard DESACTIVE: dpid={dpid} port={port_no}'})

    # ── DHCP Snooping ─────────────────────────────────────────────────────────

    @route('stp', '/stp/dhcp-snooping/trust/{dpid}/{port_no}', methods=['POST'])
    def set_dhcp_trust(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_snooping.set_trusted(dpid, port_no)
        return self._ok({
            'result': f'DHCP Trusted port: dpid={dpid} port={port_no}',
            'effect': 'DHCPOFFER/ACK/NAK autorise sur ce port uniquement',
        })

    @route('stp', '/stp/dhcp-snooping/trust/{dpid}/{port_no}', methods=['DELETE'])
    def unset_dhcp_trust(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_snooping.unset_trusted(dpid, port_no)
        return self._ok({'result': f'Port dpid={dpid} port={port_no} → UNTRUSTED'})

    @route('stp', '/dhcp-snooping/status', methods=['GET'])
    def get_dhcp_status(self, req, **kwargs):
        return self._ok(self.app.dhcp_snooping.get_status())

    @route('stp', '/stp/dhcp-snooping/alerts', methods=['GET'])
    def get_dhcp_alerts(self, req, **kwargs):
        alerts = self.app.dhcp_snooping.get_alerts()
        return self._ok({'count': len(alerts), 'alerts': alerts})

    @route('stp', '/stp/dhcp-snooping/unblock/{dpid}/{port_no}', methods=['POST'])
    def dhcp_unblock(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        if dpid not in self.app.datapaths:
            return self._err('Switch non connecte')
        self.app._do_unblock(self.app.datapaths[dpid], port_no)
        self.app.dhcp_snooping.unmark_blocked(dpid, port_no)
        return self._ok({'result': f'Port DEBLOQUE (DHCP): dpid={dpid} port={port_no}'})

    # ── DHCP Mitigation ───────────────────────────────────────────────────────

    @route('stp', '/dhcp-mitigation/status', methods=['GET'])
    def get_mitigation_status(self, req, **kwargs):
        return self._ok(self.app.dhcp_mitigation.get_full_status())

    @route('stp', '/dhcp-mitigation/threats', methods=['GET'])
    def get_threat_history(self, req, **kwargs):
        return self._ok({
            'scores':  self.app.dhcp_mitigation.threat_scorer.get_all_scores(),
            'history': self.app.dhcp_mitigation.threat_scorer.get_history(50),
        })

    @route('stp', '/dhcp-mitigation/quarantine/{dpid}/{port_no}', methods=['POST'])
    def force_quarantine(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_mitigation.quarantine_mgr.quarantine(dpid, port_no)
        return self._ok({'result': f'Quarantine forcee: dpid={dpid} port={port_no}'})

    @route('stp', '/dhcp-mitigation/release/{dpid}/{port_no}', methods=['POST'])
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

    @route('stp', '/dhcp-mitigation/reset-score/{dpid}/{port_no}', methods=['POST'])
    def reset_threat_score(self, req, dpid, port_no, **kwargs):
        dpid, port_no = int(dpid), int(port_no)
        self.app.dhcp_mitigation.reset_threat_score(dpid, port_no)
        return self._ok({'result': f'Score reinitialise: dpid={dpid} port={port_no}'})


# ── Contrôleur Principal ──────────────────────────────────────────────────────

class StandaloneSTController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {'stplib': stplib.Stp, 'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stp           = kwargs['stplib']
        self.datapaths     = {}
        self.mac_to_port   = {}
        self.port_states   = {}
        self.proto_count   = defaultdict(int)
        self.intercepted            = set()
        self._blocked_by_controller = set()
        self._inter_switch_ports    = set()
        self.pending       = PendingActions(timeout=600)
        self.security      = STPSecurityManager()
        self.dhcp_snooping = DHCPSnoopingManager()
        self.rl_graph = nx.Graph()
        self.rl_net   = SDNNetwork(self.rl_graph, mu=600, K=20, seed=None)
        self.rl_agent = QRoutingAgent(self.rl_graph, self.rl_net)
        self.rl_agent.load("controllers/qtable_geant.pkl")

        self._last_stplib_role = {}
        self._install_stplib_log_handler()

        self.stp.set_config({
        'bridge': {
        'hello_time':    1,   # détection rapide des pannes (1s au lieu de 2s)
        'forward_delay': 6,   # LISTENING+LEARNING = 6s chacun (pas 15s)
        'max_age':       10,  # BPDU staleness : 10s (pas 20s)
    }
})
        wsgi = kwargs['wsgi']
        wsgi.register(STPRestAPI, {STP_APP_KEY: self})

        self.dhcp_mitigation = DHCPMitigationLayer(self)

        # ── Ports inter-switch hardcodés (topologie statique) ─────────────────
        # SW1 (dpid=134916311342148) : eth1=port2→SW2 | eth2=port3→SW3
        # SW2 (dpid=239857082692166) : eth0=port1→SW1 | eth1=port2→SW3
        # SW3 (dpid=257854794257216) : eth0=port1→SW1 | eth1=port2→SW2
        self._inter_switch_ports = {
            (134916311342148, 2),  # SW1 eth1 → SW2
            (134916311342148, 3),  # SW1 eth2 → SW3
            (239857082692166, 1),  # SW2 eth0 → SW1
            (239857082692166, 2),  # SW2 eth1 → SW3
            (257854794257216, 1),  # SW3 eth0 → SW1
            (257854794257216, 2),  # SW3 eth1 → SW2
        }
        logger.info(f"[TOPO] Ports inter-switch (statiques): {sorted(self._inter_switch_ports)}")

        logger.info("=" * 66)
        logger.info("  [STP] Controleur SEMI-MANUEL + SECURITE + DHCP SNOOPING")
        logger.info("  STP | RSTP | MSTP | PVST | PVST+")
        logger.info("  BPDU Guard + Root Guard + DHCP Snooping + Mitigation")
        logger.info("  Edge port detection via LLDP (--observe-links requis)")
        logger.info("")
        logger.info("  GET  /stp/status          (inclut inter_switch_ports)")
        logger.info("  GET  /stp/pending")
        logger.info("  GET  /stp/security/alerts")
        logger.info("  GET  /dhcp-snooping/status")
        logger.info("  GET  /stp/dhcp-snooping/alerts")
        logger.info("  GET  /dhcp-mitigation/status")
        logger.info("  GET  /dhcp-mitigation/threats")
        logger.info("  POST /stp/confirm/<id>")
        logger.info("  POST /stp/security/bpdu-guard/<dpid>/<port>")
        logger.info("  POST /stp/security/root-guard/<dpid>/<port>")
        logger.info("  POST /stp/dhcp-snooping/trust/<dpid>/<port>")
        logger.info("  DELETE /stp/dhcp-snooping/trust/<dpid>/<port>")
        logger.info("  POST /stp/dhcp-snooping/unblock/<dpid>/<port>")
        logger.info("  POST /dhcp-mitigation/quarantine/<dpid>/<port>")
        logger.info("  POST /dhcp-mitigation/release/<dpid>/<port>")
        logger.info("  POST /dhcp-mitigation/reset-score/<dpid>/<port>")
        logger.info("=" * 66)

    def _install_stplib_log_handler(self):
        import re
        controller = self

        class StplibRoleCapture(logging.Handler):
            PATTERN = re.compile(
                r'dpid=([0-9a-fA-F]+).*\[port=(\d+)\]\s+'
                r'(ROOT_PORT|DESIGNATED_PORT|NON_DESIGNATED_PORT|DISABLED_PORT)'
            )
            ROLE_MAP = {
                'ROOT_PORT':           'ROOT',
                'DESIGNATED_PORT':     'DESIGNATED',
                'NON_DESIGNATED_PORT': 'NON_DESIGNATED',
                'DISABLED_PORT':       'DISABLED',
            }

            def emit(self, record):
                try:
                    msg = record.getMessage()
                    m   = self.PATTERN.search(msg)
                    if m:
                        dpid    = int(m.group(1), 16)
                        port_no = int(m.group(2))
                        role    = self.ROLE_MAP.get(m.group(3), 'UNKNOWN')
                        controller._last_stplib_role[(dpid, port_no)] = role
                except Exception:
                    pass

        stp_logger = logging.getLogger('ryu.lib.stplib')
        stp_logger.addHandler(StplibRoleCapture())

    # ── Switch connection ─────────────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id]   = datapath
        self.mac_to_port[datapath.id] = {}
        self.rl_graph.add_node(datapath.id)
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)
        logger.info(f"[STP] Switch connecte: dpid={datapath.id}")

    def _add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst    = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod     = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst)
        datapath.send_msg(mod)

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

        if eth.dst.lower() in (BPDU_DST_STD, BPDU_DST_PVST):
            self._handle_bpdu(datapath, in_port, msg.data)
            return

        _dhcp_pre = DHCPDetector.detect(msg.data)
        _ip4_pkt  = pkt.get_protocol(ipv4.ipv4)
        _src_ip   = _ip4_pkt.src if _ip4_pkt else ''
        if not self.dhcp_mitigation.inspect(datapath, in_port, _dhcp_pre, _src_ip, eth.src):
            return

        dhcp_info = DHCPDetector.detect(msg.data)
        if dhcp_info is not None:
            if not self._check_dhcp_snooping(datapath, in_port, dhcp_info):
                return

        self.mac_to_port[dpid][eth.src] = in_port

        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        if out_port == ofproto.OFPP_FLOOD:
            dst_dpid = None
            for sw, macs in self.mac_to_port.items():
                if sw != dpid and eth.dst in macs:
                    dst_dpid = sw
                    break
            if dst_dpid is not None:
                path = self.rl_agent.get_path(dpid, dst_dpid, lam=200, fallback=None)
                if path and len(path) >= 2:
                    next_hop = path[1]
                    learned = self.mac_to_port.get(next_hop, {})
                    for m, p in self.mac_to_port[dpid].items():
                        if m in learned:
                            out_port = p
                            break
                    logger.info(f"[RL] path {dpid}→{dst_dpid}: {path} → out_port={out_port}")

        actions  = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            self._add_flow(datapath, 1,
                           parser.OFPMatch(in_port=in_port, eth_dst=eth.dst), actions)
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data))

    # ── STP port state change ─────────────────────────────────────────────────

    @set_ev_cls(stplib.EventPortStateChange, MAIN_DISPATCHER)
    def port_state_change_handler(self, ev):
        dpid      = ev.dp.id
        port_no   = ev.port_no
        state_str = STPLIB_STATE_MAP.get(ev.port_state, 'unknown')
        protocol  = self.port_states.get((dpid, port_no), {}).get('protocol', 'STP')
        old       = self.port_states.get((dpid, port_no), {}).get('state', '?')
        port_role = self._last_stplib_role.get((dpid, port_no), 'UNKNOWN')

        logger.info(f"[{protocol}] dpid={dpid} port={port_no} "
                    f"{old.upper()} -> {PORT_ICONS.get(state_str, state_str)}  [stplib]")

        # stplib gère entièrement le blocage/déblocage STP via ses mécanismes internes.
        # Le contrôleur se contente de mettre à jour port_states pour l'API REST.
        self.port_states[(dpid, port_no)] = {
            'state':      state_str,
            'protocol':   protocol,
            'port_role':  port_role,
            'updated_at': time.strftime('%H:%M:%S'),
        }

    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def topology_change_handler(self, ev):
        self.mac_to_port[ev.dp.id] = {}

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def raw_packet_in_handler(self, ev):
        msg = ev.msg
        eth = packet.Packet(msg.data).get_protocols(ethernet.ethernet)[0]
        if eth.dst.lower() in (BPDU_DST_STD, BPDU_DST_PVST):
            self._handle_bpdu(msg.datapath, msg.match['in_port'], msg.data)

    # ── LLDP / Topology — désactivé (ports inter-switch hardcodés) ───────────
    # Les handlers EventLinkAdd / EventLinkDelete sont supprimés volontairement.
    # Lancer sans --observe-links pour éviter l'oscillation TOPO qui perturbe stplib.

    # ── DHCP Snooping ─────────────────────────────────────────────────────────

    def _check_dhcp_snooping(self, datapath, in_port, dhcp_info: dict) -> bool:
        dpid     = datapath.id
        msg_type = dhcp_info['msg_type']

        if msg_type not in DHCP_SERVER_MSG_TYPES:
            return True

        if self.dhcp_snooping.is_trusted(dpid, in_port):
            if msg_type == DHCP_ACK:
                self.dhcp_snooping.record_binding(
                    dhcp_info['client_mac'], dhcp_info['offered_ip'], dpid, in_port)
            return True

        details = (f"Rogue server: IP={dhcp_info['src_ip']} MAC={dhcp_info['src_mac']} "
                   f"MsgType={msg_type} client={dhcp_info['client_mac']}")
        logger.critical(f"!!! [DHCP SNOOPING] {details} dpid={dpid} port={in_port}")

        self.dhcp_snooping.add_alert(dpid, in_port, details)
        self.dhcp_snooping.mark_blocked(dpid, in_port)
        self._do_block(datapath, in_port, "DHCP-Snooping")

        self.port_states[(dpid, in_port)] = {
            'state': 'dhcp_rogue_err', 'protocol': 'DHCP',
            'updated_at': time.strftime('%H:%M:%S'),
            'note': 'Rogue DHCP server detected'
        }
        return False

    # ── BPDU / Security ───────────────────────────────────────────────────────

    def _handle_bpdu(self, datapath, port_no, data):
        dpid = datapath.id
        eth  = packet.Packet(data).get_protocols(ethernet.ethernet)[0]
        info = BPDUDetector.detect(eth.dst, data)
        self.proto_count[info['protocol']] += 1

        if (dpid, port_no) not in self.port_states:
            self.port_states[(dpid, port_no)] = {
                'state': 'listening', 'protocol': info['protocol'],
                'updated_at': time.strftime('%H:%M:%S')
            }

        if self.security.has_bpdu_guard(dpid, port_no):
            logger.warning(
                f"[SECURITY] BPDU Guard: BPDU recu sur port edge "
                f"dpid={dpid} port={port_no} → blocage immediat"
            )
            self.security.add_alert('BPDU_GUARD_VIOLATION', dpid, port_no,
                                    f"Received {info['protocol']} BPDU")
            self._do_block(datapath, port_no, "BPDU-Guard")
            self.port_states[(dpid, port_no)]['state'] = 'bpdu_guard_err'
            return

        if self.security.has_root_guard(dpid, port_no):
            if self.security.is_superior_bpdu(info['root_priority'], info['root_mac']):
                logger.critical(
                    f"[SECURITY] Root Guard: BPDU superieur de {info['root_id']} "
                    f"sur dpid={dpid} port={port_no} → ROOT HIJACK bloque"
                )
                self.security.add_alert('ROOT_HIJACK_ATTEMPT', dpid, port_no,
                                        f"Superior BPDU from {info['root_id']}")
                self._do_block(datapath, port_no, "Root-Guard")
                self.port_states[(dpid, port_no)]['state'] = 'root_inconsistent'
                return

        if info['root_mac'] != 'N/A' and not self.security.has_root_guard(dpid, port_no):
            self.security.update_root(info['root_priority'], info['root_mac'], dpid)

    def _propose_block(self, datapath, port, proto, reason):
        aid = self.pending.add(datapath.id, port, proto, reason)
        logger.info(f"!!! [ATTENTION] Boucle detectee ({proto})")
        logger.info(f"!!! Action requise: POST /stp/confirm/{aid} "
                    f"pour bloquer dpid={datapath.id} port={port}")

    def _do_block(self, datapath, port, protocol):
        """
        Bloque un port via FlowMod OpenFlow 1.3.
        Utilisé uniquement par : BPDU Guard, Root Guard, DHCP Snooping.

        Deux règles installées :
          prio=200  eth_dst=BPDU  → CONTROLLER   (BPDUs continuent d'arriver)
          prio=100  in_port=port  → DROP          (tout le trafic data est bloqué)
        """
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto

        self._blocked_by_controller.add((datapath.id, port))

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
            match=match_data, instructions=[])
        datapath.send_msg(mod_data)

        logger.info(
            f"[ACTION] Port BLOQUE: dpid={datapath.id} port={port} [{protocol}] "
            f"(data=DROP | BPDUs=passthrough vers controleur)"
        )

    def _do_unblock(self, datapath, port):
        """Supprime les FlowMods de blocage → port redevient normal (table-miss)."""
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto

        self._blocked_by_controller.discard((datapath.id, port))

        for bpdu_dst in ('01:80:c2:00:00:00', '01:00:0c:cc:cc:cd'):
            match_bpdu = parser.OFPMatch(in_port=port, eth_dst=bpdu_dst)
            mod_del    = parser.OFPFlowMod(
                datapath=datapath, priority=200,
                command=ofproto.OFPFC_DELETE,
                out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
                match=match_bpdu)
            datapath.send_msg(mod_del)

        match_data = parser.OFPMatch(in_port=port)
        mod_data   = parser.OFPFlowMod(
            datapath=datapath, priority=100,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY,
            match=match_data)
        datapath.send_msg(mod_data)

        logger.info(f"[ACTION] Port DEBLOQUE: dpid={datapath.id} port={port}")
