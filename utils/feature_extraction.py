"""
Feature Extraction — Ryu FlowStats → UNSW-NB15 (49 features)
============================================================
Toutes les données viennent directement de Mininet via Ryu OpenFlow.
Aucun dataset externe requis.

Pipeline:
  Mininet traffic → OVS switch → Ryu FlowStatsReply
                                → FeatureExtractor
                                → feature vector (49 cols)
                                → data/mininet_flows.csv
                                → ML training
"""

import time
import collections
import logging
import csv
import os
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Mappings ──────────────────────────────────────────────────────────────────

PROTO_MAP = {1: 'icmp', 6: 'tcp', 17: 'udp', 0: 'hopopt'}

SERVICE_MAP = {
    20: 'ftp-data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
    25: 'smtp', 53: 'dns', 67: 'dhcp', 80: 'http',
    110: 'pop3', 143: 'imap', 443: 'https', 445: 'smb',
    3306: 'mysql', 3389: 'rdp', 8080: 'http', 8443: 'https',
}

# Colonnes exactes UNSW-NB15 (ordre préservé pour le CSV)
UNSW_COLUMNS = [
    'dur', 'proto', 'service', 'state',
    'spkts', 'dpkts', 'sbytes', 'dbytes',
    'rate', 'sttl', 'dttl', 'sload', 'dload',
    'sloss', 'dloss', 'sinpkt', 'dinpkt',
    'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
    'tcprtt', 'synack', 'ackdat',
    'smean', 'dmean', 'trans_depth', 'response_body_len',
    'ct_srv_src', 'ct_dst_ltm', 'ct_src_dport_ltm',
    'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
    'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports',
    # Colonnes debug (ignorées pour ML)
    'src_ip', 'dst_ip', 'src_port', 'dst_port', 'dpid', 'timestamp',
    # Labels
    'label', 'attack_cat',
]

ML_FEATURE_COLS = [c for c in UNSW_COLUMNS
                   if c not in ('src_ip','dst_ip','src_port','dst_port',
                                'dpid','timestamp','label','attack_cat',
                                'proto','service','state')]


# ── Tracker de connexions récentes (features ct_*) ────────────────────────────

class ConnectionTracker:
    """Maintient une fenêtre glissante des connexions récentes."""

    def __init__(self, window: int = 100):
        self.window = window
        self._q: collections.deque = collections.deque(maxlen=100_000)

    def add(self, src_ip, dst_ip, src_port, dst_port, service):
        self._q.append({
            'ts': time.time(),
            'si': src_ip, 'di': dst_ip,
            'sp': src_port, 'dp': dst_port,
            'svc': service,
        })

    def _recent(self):
        cutoff = time.time() - self.window
        return [c for c in self._q if c['ts'] >= cutoff]

    def ct_srv_src(self, src_ip, service) -> int:
        return sum(1 for c in self._recent()
                   if c['si'] == src_ip and c['svc'] == service)

    def ct_dst_ltm(self, dst_ip) -> int:
        return sum(1 for c in self._recent() if c['di'] == dst_ip)

    def ct_src_dport_ltm(self, src_ip, dst_port) -> int:
        return sum(1 for c in self._recent()
                   if c['si'] == src_ip and c['dp'] == dst_port)

    def ct_dst_sport_ltm(self, dst_ip, src_port) -> int:
        return sum(1 for c in self._recent()
                   if c['di'] == dst_ip and c['sp'] == src_port)

    def ct_dst_src_ltm(self, src_ip, dst_ip) -> int:
        return sum(1 for c in self._recent()
                   if c['si'] == src_ip and c['di'] == dst_ip)

    def ct_src_ltm(self, src_ip) -> int:
        return sum(1 for c in self._recent() if c['si'] == src_ip)

    def ct_srv_dst(self, dst_ip, service) -> int:
        return sum(1 for c in self._recent()
                   if c['di'] == dst_ip and c['svc'] == service)


# ── Extracteur principal ──────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Convertit les FlowStats Ryu en vecteurs UNSW-NB15 (49 features).

    Utilisé par:
      - controller.py  →  analyse IDS en temps réel
      - collect_traffic.py  →  génération du dataset d'entraînement
    """

    def __init__(self):
        self._tracker = ConnectionTracker()
        # Snapshot précédent pour calcul des deltas
        self._prev: Dict[int, Dict[str, dict]] = {}  # {dpid: {fkey: stat}}

    # ── API principale ────────────────────────────────────────────────────────

    def extract_from_flow_stats(self, dpid: int,
                                flow_stats: list,
                                flow_history: dict = None) -> List[dict]:
        """
        Appelé depuis EventOFPFlowStatsReply.
        Retourne une liste de feature dicts.
        """
        results = []
        now = time.time()
        prev = self._prev.get(dpid, {})
        new_prev = {}

        for stat in flow_stats:
            try:
                feat = self._stat_to_features(stat, dpid, prev, now)
                if feat:
                    new_prev[feat['_fkey']] = {
                        'pkts':  stat.packet_count,
                        'bytes': stat.byte_count,
                        'ts':    now,
                    }
                    del feat['_fkey']
                    results.append(feat)
            except Exception as e:
                logger.debug(f"[FeatExtract] stat error: {e}")

        self._prev[dpid] = new_prev
        return results

    def extract_from_packet(self, pkt, ip_pkt, dpid: int, in_port: int) -> dict:
        """Extraction rapide depuis PacketIn (analyse inline temps réel)."""
        try:
            from ryu.lib.packet import tcp as tcp_mod, udp as udp_mod
            src_ip, dst_ip = ip_pkt.src, ip_pkt.dst
            proto  = PROTO_MAP.get(ip_pkt.proto, 'other')
            sp = dp = 0
            state  = 'INT'
            swin   = 0
            ttl    = ip_pkt.ttl

            tcp_p = pkt.get_protocol(tcp_mod.tcp)
            udp_p = pkt.get_protocol(udp_mod.udp)
            if tcp_p:
                sp, dp = tcp_p.src_port, tcp_p.dst_port
                swin   = getattr(tcp_p, 'window_size', 0)
                bits   = tcp_p.bits
                state  = ('REQ' if (bits & 0x02 and not bits & 0x10)
                          else 'ACC' if (bits & 0x02 and bits & 0x10)
                          else 'CLO' if bits & 0x01
                          else 'CON')
            elif udp_p:
                sp, dp = udp_p.src_port, udp_p.dst_port
                state  = 'CON'

            svc = SERVICE_MAP.get(dp, '-')
            self._tracker.add(src_ip, dst_ip, sp, dp, svc)
            pkt_len = len(pkt.data)

            return self._build_feat(
                src_ip=src_ip, dst_ip=dst_ip, sp=sp, dp=dp,
                proto=proto, service=svc, state=state,
                dur=0.0, spkts=1, dpkts=0,
                sbytes=pkt_len, dbytes=0,
                sttl=ttl, sload=0.0, smean=pkt_len, swin=swin,
                dpid=dpid,
            )
        except Exception as e:
            logger.debug(f"[FeatExtract] packet error: {e}")
            return {}

    # ── Helpers privés ────────────────────────────────────────────────────────

    def _stat_to_features(self, stat, dpid: int, prev: dict, now: float) -> Optional[dict]:
        """Convertit un OFPFlowStats en feature dict."""
        match = stat.match
        src_ip = str(match.get('ipv4_src', ''))
        dst_ip = str(match.get('ipv4_dst', ''))
        if not src_ip or src_ip == '0.0.0.0':
            return None

        proto_n = match.get('ip_proto', 0)
        proto   = PROTO_MAP.get(proto_n, 'other')
        sp      = match.get('tcp_src', match.get('udp_src', 0))
        dp      = match.get('tcp_dst', match.get('udp_dst', 0))
        svc     = SERVICE_MAP.get(dp, '-')
        fkey    = f"{src_ip}:{sp}-{dst_ip}:{dp}/{proto}"

        dur     = stat.duration_sec + stat.duration_nsec / 1e9
        dur     = max(dur, 0.001)
        pkts    = stat.packet_count
        byts    = stat.byte_count

        prev_s  = prev.get(fkey, {})
        d_pkts  = pkts - prev_s.get('pkts', 0)
        d_bytes = byts - prev_s.get('bytes', 0)
        d_t     = now - prev_s.get('ts', now - 10)
        d_t     = max(d_t, 0.001)

        rate    = pkts / dur
        sload   = (byts * 8) / dur          # bps
        sinpkt  = dur / max(pkts, 1)
        smean   = byts // max(pkts, 1)

        # État TCP simplifié
        state   = 'REQ' if pkts == 1 else 'CON'

        # Tracker update
        self._tracker.add(src_ip, dst_ip, sp, dp, svc)

        feat = self._build_feat(
            src_ip=src_ip, dst_ip=dst_ip, sp=sp, dp=dp,
            proto=proto, service=svc, state=state,
            dur=dur, spkts=pkts, dpkts=0,
            sbytes=byts, dbytes=0,
            sttl=64, sload=sload, smean=smean, swin=255,
            dpid=dpid, sinpkt=sinpkt, rate=rate,
        )
        feat['_fkey'] = fkey
        return feat

    def _build_feat(self, src_ip, dst_ip, sp, dp, proto, service,
                    state, dur, spkts, dpkts, sbytes, dbytes,
                    sttl, sload, smean, swin, dpid,
                    sinpkt=0.0, rate=0.0) -> dict:
        """Construit le feature dict complet."""
        return {
            # ── UNSW-NB15 features ────────────────────────────────────────
            'dur':      round(dur, 6),
            'proto':    proto,
            'service':  service,
            'state':    state,
            'spkts':    spkts,
            'dpkts':    dpkts,
            'sbytes':   sbytes,
            'dbytes':   dbytes,
            'rate':     round(rate, 4),
            'sttl':     sttl,
            'dttl':     0,
            'sload':    round(sload, 4),
            'dload':    0.0,
            'sloss':    0,
            'dloss':    0,
            'sinpkt':   round(sinpkt, 6),
            'dinpkt':   0.0,
            'sjit':     0.0,
            'djit':     0.0,
            'swin':     swin,
            'stcpb':    0,
            'dtcpb':    0,
            'dwin':     0,
            'tcprtt':   0.0,
            'synack':   0.0,
            'ackdat':   0.0,
            'smean':    smean,
            'dmean':    0,
            'trans_depth':          0,
            'response_body_len':    0,
            # ── Features comportementales ct_* ────────────────────────────
            'ct_srv_src':       self._tracker.ct_srv_src(src_ip, service),
            'ct_dst_ltm':       self._tracker.ct_dst_ltm(dst_ip),
            'ct_src_dport_ltm': self._tracker.ct_src_dport_ltm(src_ip, dp),
            'ct_dst_sport_ltm': self._tracker.ct_dst_sport_ltm(dst_ip, sp),
            'ct_dst_src_ltm':   self._tracker.ct_dst_src_ltm(src_ip, dst_ip),
            'is_ftp_login':     1 if service == 'ftp' else 0,
            'ct_ftp_cmd':       0,
            'ct_flw_http_mthd': 1 if service == 'http' else 0,
            'ct_src_ltm':       self._tracker.ct_src_ltm(src_ip),
            'ct_srv_dst':       self._tracker.ct_srv_dst(dst_ip, service),
            'is_sm_ips_ports':  1 if src_ip == dst_ip else 0,
            # ── Debug / metadata ──────────────────────────────────────────
            'src_ip':    src_ip,
            'dst_ip':    dst_ip,
            'src_port':  sp,
            'dst_port':  dp,
            'dpid':      dpid,
            'timestamp': time.time(),
            # ── Labels (définis par collect_traffic.py) ───────────────────
            'label':      0,
            'attack_cat': 'Normal',
        }

    # ── Export CSV ────────────────────────────────────────────────────────────

    def save_to_csv(self, features_list: List[dict],
                    csv_path: str = None, append: bool = True):
        """Écrit une liste de feature dicts dans mininet_flows.csv."""
        if not features_list:
            return
        path = csv_path or os.path.join(DATA_DIR, 'mininet_flows.csv')
        mode = 'a' if append and os.path.exists(path) else 'w'
        try:
            with open(path, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=UNSW_COLUMNS,
                                        extrasaction='ignore')
                if mode == 'w':
                    writer.writeheader()
                writer.writerows(features_list)
            logger.info(f"[FeatExtract] {len(features_list)} flows → {path}")
        except Exception as e:
            logger.error(f"[FeatExtract] CSV write error: {e}")

