"""
controller_stp.py — STP Multi-Protocole SEMI-MANUEL + Sécurité
===============================================================
Fonctionnalités:
  1. stplib détecte les boucles, on intercepte le blocage → confirmation manuelle
  2. RSTP / MSTP / PVST / PVST+ détectés via analyse BPDU raw
  3. BPDU Guard  — bloque immédiatement les ports edge qui reçoivent des BPDUs
  4. Root Guard  — empêche un port de devenir root port (Root Bridge Hijacking)

Attaque contrée — Root Bridge Hijacking:
  Un PC malveillant envoie des BPDUs avec priorité très basse + MAC faible
  pour usurper le rôle de Root Bridge et intercepter tout le trafic.

  Root Guard: activé sur les ports uplink → si un BPDU supérieur arrive,
              port passe en root-inconsistent + alerte critique
  BPDU Guard: activé sur les ports host → tout BPDU = blocage immédiat

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

Usage:
  PYTHONPATH=. ryu-manager controllers/controller_stp.py --observe-links
"""

import struct
import logging
import json
import time
import threading
from collections import defaultdict

import networkx as nx

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
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

PORT_ICONS = {
    'blocking':          '🔴 BLOCKING',
    'listening':         '🟡 LISTENING',
    'learning':          '🟠 LEARNING',
    'forwarding':        '🟢 FORWARDING',
    'discarding':        '🔴 DISCARDING',
    'disabled':          '⚫ DISABLED',
    'root_inconsistent': '🚨 ROOT-INCONSISTENT',
    'bpdu_guard_err':    '🚫 BPDU-GUARD-ERR-DISABLE',
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
    """
    BPDU Guard : port edge reçoit un BPDU → blocage immédiat.
    Root Guard : BPDU supérieur sur port protégé → root-inconsistent + alerte.
    """

    def __init__(self):
        self._lock       = threading.Lock()
        self.protected   = {}     # {(dpid, port): 'bpdu_guard'|'root_guard'|'both'}
        self.known_root  = None   # {'priority', 'mac', 'dpid', 'seen_at'}
        self.alerts      = []

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
        """Retourne True si ce BPDU est meilleur que le Root Bridge connu."""
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

    @route('stp', '/stp/status', methods=['GET'])
    def get_status(self, req, **kwargs):
        return self._ok({
            'protocols_detected': dict(self.app.proto_count),
            'port_states': {f"dpid={d}/port={p}": v
                            for (d, p), v in self.app.port_states.items()},
            'blocked_ports': [{'dpid': d, 'port': p}
                              for (d, p), v in self.app.port_states.items()
                              if v.get('state') in ('blocking', 'discarding',
                                                    'root_inconsistent', 'bpdu_guard_err')],
            'security': self.app.security.get_status(),
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

    # ── Sécurité ──────────────────────────────────────────────────────────────

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


# ── Contrôleur Principal ──────────────────────────────────────────────────────

class StandaloneSTController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {'stplib': stplib.Stp, 'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stp         = kwargs['stplib']
        self.datapaths   = {}
        self.mac_to_port = {}
        self.port_states = {}
        self.proto_count = defaultdict(int)
        self.intercepted = set()
        self.pending     = PendingActions(timeout=120)
        self.security    = STPSecurityManager()

        # Cache des rôles de port lus depuis les logs stplib
        # {(dpid, port_no): 'ROOT'|'DESIGNATED'|'NON_DESIGNATED'}
        self._last_stplib_role = {}

        # Capturer les logs stplib pour extraire les rôles de port
        self._install_stplib_log_handler()

        self.stp.set_config({
            'bridge': {'hello_time': 1, 'forward_delay': 2, 'max_age': 10}
        })
        wsgi = kwargs['wsgi']
        wsgi.register(STPRestAPI, {STP_APP_KEY: self})

        logger.info("=" * 62)
        logger.info("  [STP] Controleur SEMI-MANUEL + SECURITE")
        logger.info("  STP | RSTP | MSTP | PVST | PVST+")
        logger.info("  BPDU Guard + Root Guard actifs")
        logger.info("")
        logger.info("  GET  /stp/status")
        logger.info("  GET  /stp/pending")
        logger.info("  GET  /stp/security/alerts")
        logger.info("  POST /stp/confirm/<id>")
        logger.info("  POST /stp/security/bpdu-guard/<dpid>/<port>")
        logger.info("  POST /stp/security/root-guard/<dpid>/<port>")
        logger.info("=" * 62)

    def _install_stplib_log_handler(self):
        """
        Capture les logs stplib pour extraire les rôles de port.
        stplib logue des lignes comme:
          "[port=2] NON_DESIGNATED_PORT / BLOCK"
          "[port=1] ROOT_PORT           / LISTEN"
          "[port=3] DESIGNATED_PORT     / FORWARD"
        On parse ces lignes pour alimenter _last_stplib_role.
        """
        import re
        controller = self

        class StplibRoleCapture(logging.Handler):
            PATTERN = re.compile(
                r'dpid=([0-9a-fA-F]+).*\[port=(\d+)\]\s+'
                r'(ROOT_PORT|DESIGNATED_PORT|NON_DESIGNATED_PORT|DISABLED_PORT)'
            )
            ROLE_MAP = {
                'ROOT_PORT':          'ROOT',
                'DESIGNATED_PORT':    'DESIGNATED',
                'NON_DESIGNATED_PORT':'NON_DESIGNATED',
                'DISABLED_PORT':      'DISABLED',
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

        # Attacher au logger stplib
        stp_logger = logging.getLogger('ryu.lib.stplib')
        stp_logger.addHandler(StplibRoleCapture())

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        self.datapaths[datapath.id]   = datapath
        self.mac_to_port[datapath.id] = {}
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match   = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)
        logger.info(f"[STP] Switch connecte: dpid={datapath.id}")

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

        self.mac_to_port[dpid][eth.src] = in_port
        out_port = self.mac_to_port[dpid].get(eth.dst, ofproto.OFPP_FLOOD)
        actions  = [parser.OFPActionOutput(out_port)]
        if out_port != ofproto.OFPP_FLOOD:
            self._add_flow(datapath, 1,
                           parser.OFPMatch(in_port=in_port, eth_dst=eth.dst), actions)
        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        datapath.send_msg(parser.OFPPacketOut(
            datapath=datapath, buffer_id=msg.buffer_id,
            in_port=in_port, actions=actions, data=data))

    @set_ev_cls(stplib.EventPortStateChange, MAIN_DISPATCHER)
    def port_state_change_handler(self, ev):
        dpid      = ev.dp.id
        port_no   = ev.port_no
        state_str = STPLIB_STATE_MAP.get(ev.port_state, 'unknown')
        protocol  = self.port_states.get((dpid, port_no), {}).get('protocol', 'STP')
        old       = self.port_states.get((dpid, port_no), {}).get('state', '?')

        logger.info(f"[{protocol}] dpid={dpid} port={port_no} "
                    f"{old.upper()} -> {PORT_ICONS.get(state_str, state_str)}  [stplib]")

        if ev.port_state == stplib.PORT_STATE_BLOCK:
            # Vrai NON_DESIGNATED : LEARNING -> BLOCK (convergence terminee)
            # Transitoire         : LISTENING -> BLOCK (convergence en cours)
            # old_state est fiable car le log stplib arrive en meme temps
            # que l evenement -> race condition sur _last_stplib_role
            is_final_block = (old == 'learning')
            port_role = self._last_stplib_role.get((dpid, port_no), 'UNKNOWN')

            self.port_states[(dpid, port_no)] = {
                'state':      'pending_block',
                'protocol':   protocol,
                'port_role':  port_role,
                'updated_at': time.strftime('%H:%M:%S'),
                'note':       'NON_DESIGNATED final' if is_final_block else 'BLOCK transitoire',
            }

            if is_final_block:
                self.intercepted.add((dpid, port_no))
                if not self.pending.already_pending(dpid, port_no):
                    self._propose_block(
                        ev.dp, port_no, protocol,
                        f"Port NON_DESIGNATED (LEARNING->BLOCK): dpid={dpid} port={port_no}")
            else:
                logger.debug(f"[STP] Transitoire ignore (old={old}): dpid={dpid} port={port_no}")
            return

        # ── Autres états → appliquer ──────────────────────────────────
        self.port_states[(dpid, port_no)] = {
            'state': state_str, 'protocol': protocol,
            'updated_at': time.strftime('%H:%M:%S'),
        }
        if (dpid, port_no) in self.intercepted and state_str == 'forwarding':
            self.intercepted.discard((dpid, port_no))

    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def topology_change_handler(self, ev):
        self.mac_to_port[ev.dp.id] = {}

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def raw_packet_in_handler(self, ev):
        msg = ev.msg
        eth = packet.Packet(msg.data).get_protocols(ethernet.ethernet)[0]
        if eth.dst.lower() in (BPDU_DST_STD, BPDU_DST_PVST):
            self._handle_bpdu(msg.datapath, msg.match['in_port'], msg.data)

    def _handle_bpdu(self, datapath, in_port, raw):
        eth      = packet.Packet(raw).get_protocols(ethernet.ethernet)[0]
        info     = BPDUDetector.detect(eth.dst, raw)
        protocol = info['protocol']
        dpid     = datapath.id

        if protocol == 'UNKNOWN':
            return

        self.proto_count[protocol] += 1

        # ── 1. BPDU Guard ─────────────────────────────────────────────
        if self.security.has_bpdu_guard(dpid, in_port):
            self._trigger_bpdu_guard(datapath, in_port, protocol, info)
            return

        # ── 2. Root Guard ─────────────────────────────────────────────
        if self.security.has_root_guard(dpid, in_port):
            if self.security.is_superior_bpdu(info['root_priority'], info['root_mac']):
                self._trigger_root_guard(datapath, in_port, protocol, info)
                return

        # ── 3. Mettre à jour Root Bridge connu ────────────────────────
        if (info['root_priority'] < MAX_PRIORITY and info['root_mac'] != 'N/A'):
            known = self.security.known_root
            if (known is None or
                    info['root_priority'] < known.get('priority', MAX_PRIORITY) or
                    (info['root_priority'] == known.get('priority') and
                     info['root_mac'] < known.get('mac', 'ff:ff:ff:ff:ff:ff'))):
                self.security.update_root(info['root_priority'], info['root_mac'], dpid)
                logger.info(f"[STP] Root Bridge connu: prio={info['root_priority']} "
                            f"mac={info['root_mac']}")

        if protocol == 'STP':
            return

        # ── 4. État du port (RSTP/MSTP/PVST/PVST+) ───────────────────
        flags = info['flags']
        state = ('forwarding' if flags & FLAG_FORWARDING else
                 'learning'   if flags & FLAG_LEARNING   else 'discarding')

        old = self.port_states.get((dpid, in_port), {}).get('state', '?')
        if old != state:
            logger.info(f"[{protocol}] dpid={dpid} port={in_port} "
                        f"{old.upper()} -> {PORT_ICONS.get(state, state)}")

        self.port_states[(dpid, in_port)] = {
            'state': state, 'protocol': protocol,
            'root_id': info['root_id'], 'bridge_id': info['bridge_id'],
            'vlan': info['vlan'], 'instance': info['instance'],
            'updated_at': time.strftime('%H:%M:%S'),
        }

        if state == 'discarding' and (flags & FLAG_PROPOSAL):
            if not self.pending.already_pending(dpid, in_port):
                self._propose_block(datapath, in_port, protocol,
                                    f"PROPOSAL BPDU {protocol} recu")

    # ── BPDU Guard ────────────────────────────────────────────────────────────

    def _trigger_bpdu_guard(self, datapath, port_no, protocol, info):
        dpid = datapath.id
        self._do_block(datapath, port_no, protocol, state='bpdu_guard_err')
        self.security.add_alert('BPDU_GUARD_TRIGGERED', dpid, port_no, {
            'protocol':  protocol,
            'bridge_id': info['bridge_id'],
            'root_id':   info['root_id'],
            'message':   'BPDU recu sur port edge — possible attaque STP',
        })
        logger.warning("")
        logger.warning("=" * 62)
        logger.warning(f"  🚫 BPDU GUARD — PORT BLOQUE IMMEDIATEMMENT")
        logger.warning(f"  dpid={dpid} port={port_no} [{protocol}]")
        logger.warning(f"  BPDU recu sur port host — comportement anormal")
        logger.warning(f"  Bridge: {info['bridge_id']}  Root: {info['root_id']}")
        logger.warning(f"  Debloquer: curl -X POST http://localhost:8080/stp/unblock/{dpid}/{port_no}")
        logger.warning(f"  Alertes : curl http://localhost:8080/stp/security/alerts")
        logger.warning("=" * 62)
        logger.warning("")

    # ── Root Guard ────────────────────────────────────────────────────────────

    def _trigger_root_guard(self, datapath, port_no, protocol, info):
        dpid = datapath.id
        self._do_block(datapath, port_no, protocol, state='root_inconsistent')
        self.security.add_alert('ROOT_BRIDGE_HIJACK_ATTEMPT', dpid, port_no, {
            'protocol':           protocol,
            'attacker_root_id':   info['root_id'],
            'attacker_bridge_id': info['bridge_id'],
            'attacker_mac':       info['root_mac'],
            'attacker_priority':  info['root_priority'],
            'known_root':         self.security.known_root,
            'message':            'TENTATIVE ROOT BRIDGE HIJACKING',
        })
        logger.warning("")
        logger.warning("█" * 62)
        logger.warning(f"  🚨 ROOT BRIDGE HIJACKING DETECTE !")
        logger.warning(f"  dpid={dpid} port={port_no} [{protocol}]")
        logger.warning(f"  Attaquant Root    : {info['root_id']}")
        logger.warning(f"  Attaquant Bridge  : {info['bridge_id']}")
        logger.warning(f"  Attaquant Priorite: {info['root_priority']}")
        logger.warning(f"  Attaquant MAC     : {info['root_mac']}")
        logger.warning(f"  Root Bridge connu : {self.security.known_root}")
        logger.warning(f"  Port -> ROOT-INCONSISTENT")
        logger.warning(f"  Alertes: curl http://localhost:8080/stp/security/alerts")
        logger.warning("█" * 62)
        logger.warning("")

    # ── Lecture du rôle de port stplib ───────────────────────────────────────

    def _get_port_role(self, bridge, port_no) -> str:
        """
        Lit le rôle du port depuis stplib.Bridge.
        NON_DESIGNATED = seul port à bloquer définitivement.
        DESIGNATED / ROOT en BLOCK = états transitoires de convergence.
        """
        try:
            # Constantes de rôle stplib
            ROLE_ROOT          = getattr(stplib, 'ROLE_ROOT',          1)
            ROLE_DESIGNATED    = getattr(stplib, 'ROLE_DESIGNATED',    2)
            ROLE_NON_DESIGNATED= getattr(stplib, 'ROLE_NON_DESIGNATED',3)
            ROLE_DISABLE       = getattr(stplib, 'ROLE_DISABLE',       4)

            role_map = {
                ROLE_ROOT:           'ROOT',
                ROLE_DESIGNATED:     'DESIGNATED',
                ROLE_NON_DESIGNATED: 'NON_DESIGNATED',
                ROLE_DISABLE:        'DISABLED',
            }

            # stplib.Bridge stocke les ports dans bridge.ports
            if hasattr(bridge, 'ports'):
                port_obj = bridge.ports.get(port_no)
                if port_obj is not None:
                    role = (getattr(port_obj, 'role', None) or
                            getattr(port_obj, 'port_role', None))
                    if role is not None:
                        return role_map.get(role, f'UNKNOWN({role})')

            # Fallback: log stplib intercepté dans _last_stplib_role
            return self._last_stplib_role.get((getattr(bridge, 'dpid', 0), port_no),
                                               'UNKNOWN')
        except Exception as e:
            logger.debug(f"[STP] _get_port_role error: {e}")
            return 'UNKNOWN'

    # ── Blocage semi-manuel ───────────────────────────────────────────────────

    def _propose_block(self, datapath, port_no, protocol, reason):
        dpid = datapath.id
        aid  = self.pending.add(dpid, port_no, protocol, reason)
        logger.warning("")
        logger.warning("=" * 62)
        logger.warning(f"  ⚠️  BLOCAGE EN ATTENTE DE CONFIRMATION  #{aid}")
        logger.warning(f"  Protocole : {protocol}  Switch: dpid={dpid}  Port: {port_no}")
        logger.warning(f"  Raison    : {reason}")
        logger.warning(f"  CONFIRMER : curl -X POST http://localhost:8080/stp/confirm/{aid}")
        logger.warning(f"  ANNULER   : curl -X POST http://localhost:8080/stp/cancel/{aid}")
        logger.warning(f"  Timeout   : 120s")
        logger.warning("=" * 62)
        logger.warning("")

    def _do_block(self, datapath, port_no, protocol, state=None):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=300,
            command=ofproto.OFPFC_ADD,
            match=parser.OFPMatch(in_port=port_no),
            instructions=[])
        datapath.send_msg(mod)
        if state is None:
            state = 'blocking' if protocol in ('STP', 'PVST') else 'discarding'
        self.port_states[(datapath.id, port_no)] = {
            'state': state, 'protocol': protocol,
            'updated_at': time.strftime('%H:%M:%S'),
        }
        self.intercepted.discard((datapath.id, port_no))
        logger.warning(f"[{protocol}] Port BLOQUE: dpid={datapath.id} "
                       f"port={port_no} -> {PORT_ICONS.get(state, state)}")

    def _do_unblock(self, datapath, port_no):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=300,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
            match=parser.OFPMatch(in_port=port_no))
        datapath.send_msg(mod)
        key = (datapath.id, port_no)
        if key in self.port_states:
            self.port_states[key].update({'state': 'forwarding',
                                          'updated_at': time.strftime('%H:%M:%S')})
        self.intercepted.discard(key)
        logger.info(f"[STP] Port DEBLOQUE: dpid={datapath.id} port={port_no} "
                    f"-> {PORT_ICONS['forwarding']}")

    def _add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser  = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        datapath.send_msg(parser.OFPFlowMod(
            datapath=datapath, priority=priority,
            match=match, instructions=inst))
