"""
stp_core.py — Simple Switch STP + protection Root Hijack PRÉVENTIVE
====================================================================
Fix principal : pendant la phase d'apprentissage, un seul candidat Root
est autorisé. Tout autre MAC est droppé → empêche l'oscillation STP
et la boucle infinie de TCN.

Architecture :
  BPDU entrant → [EventOFPPacketIn RAW] → DROP si suspect/attaque
                                        → laisser passer si légitime → stplib

REST API :
  GET  /stp/root
  GET  /stp/pending
  GET  /stp/alerts
  POST /stp/confirm/<id>  — accepter le nouveau Root
  POST /stp/reject/<id>   — rejeter définitivement
  POST /stp/set-root      — déclarer Root manuellement

Usage :
  ryu-manager stp_core.py --observe-links
"""

import struct
import logging
import json
import time
import threading
from collections import defaultdict

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet
from ryu.lib import stplib
from ryu.app.wsgi import ControllerBase, WSGIApplication, route
from webob import Response

logger = logging.getLogger(__name__)

ROOT_CONFIRM_THRESHOLD = 5

BPDU_DST_STD  = '01:80:c2:00:00:00'
BPDU_DST_PVST = '01:00:0c:cc:cc:cd'
BPDU_DSTS     = (BPDU_DST_STD, BPDU_DST_PVST)


# ══════════════════════════════════════════════════════════════════════════════
#  Parser BPDU
# ══════════════════════════════════════════════════════════════════════════════

def parse_bpdu_root(raw: bytes):
    try:
        offset = 14
        if len(raw) > offset and raw[offset] == 0x42 and raw[offset+1] == 0x42:
            offset += 3
        elif len(raw) > offset and raw[offset:offset+3] == b'\xaa\xaa\x03':
            offset += 8
        else:
            return None
        if len(raw) < offset + 35:
            return None
        if raw[offset + 3] != 0x00:
            return None
        root_raw = raw[offset + 5: offset + 13]
        if len(root_raw) < 8:
            return None
        priority = struct.unpack('!H', root_raw[:2])[0]
        mac      = ':'.join(f'{b:02x}' for b in root_raw[2:8])
        return {'priority': priority, 'mac': mac}
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Moniteur Root Bridge
# ══════════════════════════════════════════════════════════════════════════════

class RootBridgeMonitor:
    """
    Classifie chaque BPDU reçu AVANT qu'il atteigne stplib.

    Phase apprentissage (confirmed=False) :
      - Le PREMIER MAC non-suspect devient le candidat unique.
      - Tout autre MAC est droppé → stplib ne voit qu'un seul Root candidat
        → pas d'oscillation → pas de boucle TCN.
      - Après threshold BPDUs du même MAC → Root confirmé.

    Phase post-confirmation (confirmed=True) :
      - Même MAC → pass
      - MAC différent + meilleure priorité → drop_attack + alerte
      - MAC différent + moins bonne priorité → pass (switch secondaire)
    """

    def __init__(self, threshold=ROOT_CONFIRM_THRESHOLD):
        self._lock              = threading.Lock()
        self.threshold          = threshold
        self.known_root         = None
        self._candidates        = defaultdict(int)
        self.confirmed          = False
        self._current_candidate = None   # FIX : un seul candidat pendant l'apprentissage
        self._allowed_macs      = set()  # MACs explicitement autorisés après confirm

    def is_intrinsically_suspicious(self, priority, mac):
        """
        priority=0 et MAC=00:00:00:00:00:00 sont les valeurs classiques
        utilisées dans les attaques Root Hijack. Aucun switch réel ne les émet.
        """
        if priority == 0:
            return True
        if mac == '00:00:00:00:00:00':
            return True
        if mac == 'ff:ff:ff:ff:ff:ff':
            return True
        return False

    def classify(self, priority, mac):
        """
        Retourne :
          'drop_suspicious' — priority=0 ou MAC réservé → DROP immédiat
          'drop_attack'     — meilleure priorité que Root connu → DROP + alerte
          'learning'        — apprentissage en cours (candidat unique accepté)
          'pass'            — Root légitime → laisser passer stplib
        """
        # Niveau 1 : suspect intrinsèque → DROP immédiat
        if self.is_intrinsically_suspicious(priority, mac):
            return 'drop_suspicious'

        with self._lock:
            # MAC explicitement autorisé (après confirm manuel)
            if mac in self._allowed_macs:
                return 'pass'

            # ── Phase apprentissage ───────────────────────────────────────
            if not self.confirmed:

                # FIX PRINCIPAL :
                # Si on a déjà un candidat et que ce MAC est différent → DROP.
                # stplib ne voit qu'un seul Root pendant l'apprentissage.
                # Ça empêche l'oscillation entre deux Root candidats
                # et la boucle infinie de TCN qui en résulte.
                if self._current_candidate is not None and mac != self._current_candidate:
                    return 'drop_attack'

                # Premier BPDU non-suspect → il devient le candidat unique
                self._current_candidate = mac
                self._candidates[mac] += 1
                count = self._candidates[mac]

                if count >= self.threshold:
                    self.known_root = {
                        'priority':     priority,
                        'mac':          mac,
                        'count':        count,
                        'confirmed_at': time.strftime('%H:%M:%S'),
                    }
                    self.confirmed = True
                    self._allowed_macs.add(mac)
                    logger.info(
                        f"[ROOT] ✅ Root CONFIRMÉ après {count} BPDUs : "
                        f"priority={priority} mac={mac}"
                    )
                    return 'pass'

                return 'learning'

            # ── Phase post-confirmation ───────────────────────────────────
            if mac == self.known_root['mac']:
                self.known_root['count'] += 1
                return 'pass'

            # MAC différent avec meilleure priorité → attaque potentielle
            if self._is_better(priority, mac,
                                self.known_root['priority'], self.known_root['mac']):
                return 'drop_attack'

            # MAC différent mais moins bon → switch secondaire, on ignore
            return 'pass'

    def _is_better(self, prio_new, mac_new, prio_known, mac_known):
        return (prio_new < prio_known or
                (prio_new == prio_known and mac_new < mac_known))

    def allow_mac(self, priority, mac):
        """Après confirm manuel : autoriser ce MAC comme nouveau Root."""
        with self._lock:
            self._allowed_macs.add(mac)
            self.known_root = {
                'priority':     priority,
                'mac':          mac,
                'count':        self.threshold,
                'confirmed_at': time.strftime('%H:%M:%S') + ' (confirmé manuellement)',
            }
            self.confirmed = True
            self._candidates[mac] = self.threshold

    def force_set(self, priority, mac):
        """Déclaration manuelle du Root via REST."""
        with self._lock:
            self._allowed_macs.add(mac)
            self._current_candidate = mac
            self.known_root = {
                'priority':     priority,
                'mac':          mac,
                'count':        0,
                'confirmed_at': time.strftime('%H:%M:%S') + ' (manuel)',
            }
            self.confirmed = True
            self._candidates[mac] = self.threshold

    def get_status(self):
        with self._lock:
            return {
                'confirmed':          self.confirmed,
                'known_root':         self.known_root,
                'threshold':          self.threshold,
                'current_candidate':  self._current_candidate,
                'candidates':         dict(self._candidates),
                'allowed_macs':       list(self._allowed_macs),
            }


# ══════════════════════════════════════════════════════════════════════════════
#  File d'alertes
# ══════════════════════════════════════════════════════════════════════════════

class PendingAlerts:
    def __init__(self, timeout=300):
        self._lock    = threading.Lock()
        self._alerts  = {}
        self._counter = 0
        self.timeout  = timeout

    def add(self, dpid, port, new_priority, new_mac, known_root, reason):
        with self._lock:
            # Pas de doublon pour le même MAC
            if any(v['new_mac'] == new_mac for v in self._alerts.values()):
                return None
            self._counter += 1
            aid = self._counter
            self._alerts[aid] = {
                'id':           aid,
                'dpid':         dpid,
                'port':         port,
                'new_priority': new_priority,
                'new_mac':      new_mac,
                'known_root':   known_root,
                'reason':       reason,
                'created_at':   time.time(),
                'timestamp':    time.strftime('%H:%M:%S'),
            }
            severity = '🚨 CRITIQUE' if reason == 'SUSPICIOUS_ROOT' else '⚠️  ATTAQUE'
            logger.warning(
                f"\n{'='*62}\n"
                f"  {severity} — #{aid} {reason}\n"
                f"  Port source   : dpid={dpid} port={port}\n"
                f"  Root reçu     : priority={new_priority}  mac={new_mac}\n"
                f"  Root connu    : {known_root}\n"
                f"  BPDUs droppés : oui (stplib protégé)\n"
                f"  → POST /stp/confirm/{aid}  pour AUTORISER\n"
                f"  → POST /stp/reject/{aid}   pour REJETER\n"
                f"{'='*62}"
            )
            return aid

    def pop(self, aid):
        with self._lock:
            return self._alerts.pop(aid, None)

    def all(self):
        with self._lock:
            now = time.time()
            for k in [k for k, v in self._alerts.items()
                      if now - v['created_at'] > self.timeout]:
                del self._alerts[k]
            return [
                {**v, 'remaining_s': int(self.timeout - (now - v['created_at']))}
                for v in self._alerts.values()
            ]


# ══════════════════════════════════════════════════════════════════════════════
#  REST API
# ══════════════════════════════════════════════════════════════════════════════

APP_KEY = 'stp_app'


class STPRestAPI(ControllerBase):
    def __init__(self, req, link, data, **config):
        super().__init__(req, link, data, **config)
        self.app = data[APP_KEY]

    def _json(self, data, status=200):
        return Response(
            status=status, content_type='application/json', charset='utf-8',
            body=json.dumps(data, indent=2, default=str).encode('utf-8'),
        )

    @route('stp', '/stp/root', methods=['GET'])
    def get_root(self, req, **_):
        return self._json(self.app.monitor.get_status())

    @route('stp', '/stp/alerts', methods=['GET'])
    def get_alerts(self, req, **_):
        return self._json({'alerts': self.app.security_alerts[-50:][::-1]})

    @route('stp', '/stp/pending', methods=['GET'])
    def get_pending(self, req, **_):
        items = self.app.pending.all()
        return self._json({
            'count':   len(items),
            'pending': items,
            'note':    'BPDUs suspects droppés pendant l\'attente.',
        })

    @route('stp', '/stp/confirm/{alert_id}', methods=['POST'])
    def confirm(self, req, alert_id, **_):
        alert = self.app.pending.pop(int(alert_id))
        if not alert:
            return self._json({'error': 'Introuvable ou expirée'}, 404)
        self.app.monitor.allow_mac(alert['new_priority'], alert['new_mac'])
        self.app._log_alert('ROOT_CHANGE_ACCEPTED', alert)
        return self._json({
            'result': f"✅ Root autorisé : {alert['new_mac']}",
            'note':   'Ses BPDUs passeront désormais à stplib.',
        })

    @route('stp', '/stp/reject/{alert_id}', methods=['POST'])
    def reject(self, req, alert_id, **_):
        alert = self.app.pending.pop(int(alert_id))
        if not alert:
            return self._json({'error': 'Introuvable ou expirée'}, 404)
        self.app._log_alert('ROOT_REJECTED', alert)
        return self._json({
            'result': f"🚫 Root rejeté : {alert['new_mac']}",
            'note':   'Ses BPDUs continueront d\'être droppés.',
        })

    @route('stp', '/stp/set-root', methods=['POST'])
    def set_root(self, req, **_):
        try:
            body = json.loads(req.body)
            self.app.monitor.force_set(int(body['priority']), str(body['mac']))
            return self._json({'result': 'Root déclaré manuellement'})
        except Exception:
            return self._json(
                {'error': '{"priority": 32768, "mac": "aa:bb:cc:dd:ee:ff"}'}, 400)


# ══════════════════════════════════════════════════════════════════════════════
#  Contrôleur principal
# ══════════════════════════════════════════════════════════════════════════════

class STPSecureSwitch(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    _CONTEXTS    = {'stplib': stplib.Stp, 'wsgi': WSGIApplication}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stp             = kwargs['stplib']
        self.mac_to_port     = {}
        self.datapaths       = {}
        self.monitor         = RootBridgeMonitor(threshold=ROOT_CONFIRM_THRESHOLD)
        self.pending         = PendingAlerts(timeout=300)
        self.security_alerts = []
        self._rejected_macs  = set()

        self.stp.set_config({'bridge': {
            'hello_time': 2, 'forward_delay': 15, 'max_age': 20,
        }})
        kwargs['wsgi'].register(STPRestAPI, {APP_KEY: self})

        logger.info("=" * 62)
        logger.info("  [STP] Simple Switch STP — Protection Root Hijack PRÉVENTIVE")
        logger.info(f"  Root légitime après {ROOT_CONFIRM_THRESHOLD} BPDUs")
        logger.info("  Un seul candidat Root pendant l'apprentissage (anti-TCN)")
        logger.info("  BPDUs suspects → DROP avant stplib")
        logger.info("  priority=0 ou MAC=00:00:... → DROP immédiat")
        logger.info("  Root différent → DROP + alerte + attente manuelle")
        logger.info("")
        logger.info("  GET  /stp/root | /stp/pending | /stp/alerts")
        logger.info("  POST /stp/confirm/<id>  ← autoriser nouveau Root")
        logger.info("  POST /stp/reject/<id>   ← rejeter définitivement")
        logger.info("  POST /stp/set-root      ← déclarer Root manuellement")
        logger.info("=" * 62)

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        dp      = ev.msg.datapath
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        self.datapaths[dp.id]   = dp
        self.mac_to_port[dp.id] = {}
        self._add_flow(dp, 0, parser.OFPMatch(),
                       [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                               ofproto.OFPCML_NO_BUFFER)])
        logger.info(f"[CONNECT] dpid={dp.id}")

    @set_ev_cls(stplib.EventPortStateChange, MAIN_DISPATCHER)
    def port_state_change(self, ev):
        icons = {
            stplib.PORT_STATE_DISABLE: '⚫ DISABLED',
            stplib.PORT_STATE_BLOCK:   '🔴 BLOCKING',
            stplib.PORT_STATE_LISTEN:  '🟡 LISTENING',
            stplib.PORT_STATE_LEARN:   '🟠 LEARNING',
            stplib.PORT_STATE_FORWARD: '🟢 FORWARDING',
        }
        logger.info(f"[STP] dpid={ev.dp.id} port={ev.port_no} "
                    f"→ {icons.get(ev.port_state, '?')}")

    @set_ev_cls(stplib.EventTopologyChange, MAIN_DISPATCHER)
    def topology_change(self, ev):
        # Vider la table MAC silencieusement (sans log pour ne pas spammer)
        self.mac_to_port.get(ev.dp.id, {}).clear()

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        msg     = ev.msg
        dp      = msg.datapath
        ofproto = dp.ofproto
        parser  = dp.ofproto_parser
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]

        # ── BPDU : filtrage préventif AVANT stplib ────────────────────────
        if eth.dst.lower() in BPDU_DSTS:
            root = parse_bpdu_root(msg.data)
            if root is not None:
                decision = self.monitor.classify(root['priority'], root['mac'])

                if decision == 'drop_suspicious':
                    self._drop(dp, msg)
                    if root['mac'] not in self._rejected_macs:
                        self._rejected_macs.add(root['mac'])
                        self.pending.add(
                            dpid=dp.id, port=in_port,
                            new_priority=root['priority'],
                            new_mac=root['mac'],
                            known_root=self.monitor.known_root or 'non confirmé',
                            reason='SUSPICIOUS_ROOT',
                        )
                        self._log_alert('SUSPICIOUS_ROOT_DETECTED', {
                            'dpid': dp.id, 'port': in_port,
                            'mac': root['mac'], 'priority': root['priority'],
                        })
                    return

                elif decision == 'drop_attack':
                    self._drop(dp, msg)
                    aid = self.pending.add(
                        dpid=dp.id, port=in_port,
                        new_priority=root['priority'],
                        new_mac=root['mac'],
                        known_root=self.monitor.known_root,
                        reason='ROOT_HIJACK',
                    )
                    if aid:
                        self._log_alert('ROOT_HIJACK_ATTEMPT', {
                            'dpid': dp.id, 'port': in_port,
                            'new_mac': root['mac'],
                            'new_priority': root['priority'],
                            'known_root': self.monitor.known_root,
                        })
                    return

                elif decision == 'learning':
                    count = self.monitor.get_status()['candidates'].get(root['mac'], 0)
                    logger.info(
                        f"[ROOT] Apprentissage : {root['mac']} "
                        f"({count}/{self.monitor.threshold})"
                    )
                    # Laisser passer → stplib apprend ce candidat unique
                    return

                # decision == 'pass' → ne rien faire, stplib traite
            return

        # ── Trafic data normal ────────────────────────────────────────────
        dpid = dp.id
        self.mac_to_port.setdefault(dpid, {})
        self.mac_to_port[dpid][eth.src] = in_port

        if eth.dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][eth.dst]
            match    = parser.OFPMatch(in_port=in_port, eth_dst=eth.dst)
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self._add_flow(dp, 1, match,
                               [parser.OFPActionOutput(out_port)],
                               buffer_id=msg.buffer_id)
                return
            self._add_flow(dp, 1, match, [parser.OFPActionOutput(out_port)])
        else:
            out_port = ofproto.OFPP_FLOOD

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(
            datapath=dp, buffer_id=msg.buffer_id, in_port=in_port,
            actions=[parser.OFPActionOutput(out_port)], data=data,
        ))

    def _drop(self, dp, msg):
        """PacketOut sans actions = DROP. stplib ne verra pas ce paquet."""
        parser  = dp.ofproto_parser
        ofproto = dp.ofproto
        data    = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        dp.send_msg(parser.OFPPacketOut(
            datapath=dp,
            buffer_id=msg.buffer_id,
            in_port=msg.match['in_port'],
            actions=[],
            data=data,
        ))

    def _log_alert(self, alert_type, details):
        self.security_alerts.append({
            'type':      alert_type,
            'details':   details,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        self.security_alerts = self.security_alerts[-200:]

    def _add_flow(self, dp, priority, match, actions, buffer_id=None):
        parser  = dp.ofproto_parser
        ofproto = dp.ofproto
        inst    = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        kwargs  = dict(datapath=dp, priority=priority, match=match, instructions=inst)
        if buffer_id is not None and buffer_id != ofproto.OFP_NO_BUFFER:
            kwargs['buffer_id'] = buffer_id
        dp.send_msg(parser.OFPFlowMod(**kwargs))
