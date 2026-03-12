"""
dhcp_mitigation.py — Mitigation avancée contre les attaques Rogue DHCP Server
===============================================================================
Ce module s'ajoute à controller_stp.py SANS modifier le code existant.
Il s'injecte via monkey-patching post-instanciation dans __init__ du contrôleur.

Couches de mitigation ajoutées :
  1. Rate Limiting DHCP       — bloque un port qui envoie trop de paquets DHCP/s
  2. IP Source Guard          — vérifie cohérence MAC↔IP après binding
  3. Auto-Quarantine          — cooldown configurable avant déblocage
  4. Threat Scoring           — score cumulatif par port → quarantine permanente

Intégration (ajouter à la fin de StandaloneSTController.__init__) :
  from dhcp_mitigation import DHCPMitigationLayer
  self.dhcp_mitigation = DHCPMitigationLayer(self)

Nouveaux endpoints REST (ajouter dans STPRestAPI) :
  GET  /dhcp-mitigation/status
  GET  /dhcp-mitigation/threats
  POST /dhcp-mitigation/quarantine/<dpid>/<port>
  POST /dhcp-mitigation/release/<dpid>/<port>
  POST /dhcp-mitigation/reset-score/<dpid>/<port>

Usage :
  Le layer s'active en appelant dhcp_mitigation.inspect(datapath, in_port, dhcp_info)
  AVANT _check_dhcp_snooping dans packet_in_handler.
"""

import time
import threading
import logging
from collections import defaultdict
from typing import Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

RATE_LIMIT_WINDOW_S     = 5      # fenêtre glissante en secondes
RATE_LIMIT_MAX_PKTS     = 10     # max paquets DHCP par fenêtre par port
QUARANTINE_COOLDOWN_S   = 300    # 5 min avant déblocage autorisé
THREAT_SCORE_BLOCK      = 3      # score pour blocage automatique
THREAT_SCORE_PERMANENT  = 10     # score pour quarantine permanente

# ── Couche 1 : Rate Limiter DHCP ─────────────────────────────────────────────

class DHCPRateLimiter:
    """
    Fenêtre glissante par (dpid, port).
    Si un port envoie plus de RATE_LIMIT_MAX_PKTS paquets DHCP
    dans RATE_LIMIT_WINDOW_S secondes → violation signalée.
    """

    def __init__(self):
        self._lock     = threading.Lock()
        # {(dpid, port): [timestamp, timestamp, ...]}
        self._windows  = defaultdict(list)

    def record_and_check(self, dpid: int, port: int) -> bool:
        """
        Enregistre un paquet DHCP et retourne True si la limite est dépassée.
        """
        key = (dpid, port)
        now = time.time()
        with self._lock:
            # Garder uniquement les timestamps dans la fenêtre courante
            self._windows[key] = [
                t for t in self._windows[key]
                if now - t < RATE_LIMIT_WINDOW_S
            ]
            self._windows[key].append(now)
            count = len(self._windows[key])

        if count > RATE_LIMIT_MAX_PKTS:
            logger.warning(
                f"[RATE-LIMIT] dpid={dpid} port={port} → "
                f"{count} paquets DHCP en {RATE_LIMIT_WINDOW_S}s (max={RATE_LIMIT_MAX_PKTS})"
            )
            return True
        return False

    def reset(self, dpid: int, port: int):
        with self._lock:
            self._windows.pop((dpid, port), None)

    def get_stats(self) -> dict:
        with self._lock:
            now = time.time()
            return {
                f"dpid={d}/port={p}": len([
                    t for t in ts if now - t < RATE_LIMIT_WINDOW_S
                ])
                for (d, p), ts in self._windows.items()
            }


# ── Couche 2 : IP Source Guard ────────────────────────────────────────────────

class IPSourceGuard:
    """
    Après qu'un DHCPACK est enregistré dans la binding table,
    tout paquet IP d'une MAC avec une IP différente de celle assignée
    est considéré suspect (IP Spoofing / attaque post-binding).

    S'appuie sur dhcp_snooping.binding existant — lecture seule.
    """

    def __init__(self, binding_ref: dict, lock_ref: threading.Lock):
        """
        binding_ref  : référence directe à DHCPSnoopingManager.binding
        lock_ref     : référence directe au _lock de DHCPSnoopingManager
        """
        self._binding = binding_ref
        self._lock    = lock_ref
        self._violations = []   # log des violations

    def check(self, src_mac: str, src_ip: str, dpid: int, port: int) -> bool:
        """
        Retourne True si la combinaison MAC+IP est cohérente avec la binding table.
        Retourne False si incohérence détectée (violation).
        Si la MAC n'est pas dans la table → on laisse passer (pas encore bindée).
        """
        with self._lock:
            entry = self._binding.get(src_mac)

        if entry is None:
            return True  # MAC inconnue, on ne peut pas juger

        expected_ip = entry.get('ip')
        if expected_ip and src_ip != expected_ip and src_ip != '0.0.0.0':
            violation = {
                'src_mac':     src_mac,
                'src_ip':      src_ip,
                'expected_ip': expected_ip,
                'dpid':        dpid,
                'port':        port,
                'timestamp':   time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            self._violations.append(violation)
            self._violations = self._violations[-200:]
            logger.critical(
                f"[IP SOURCE GUARD] MAC={src_mac} utilise IP={src_ip} "
                f"mais binding dit {expected_ip} — dpid={dpid} port={port}"
            )
            return False
        return True

    def get_violations(self) -> list:
        return list(reversed(self._violations))


# ── Couche 3 : Auto-Quarantine avec cooldown ──────────────────────────────────

class QuarantineManager:
    """
    Un port bloqué entre en quarantine avec un timestamp.
    _do_unblock ne sera pas autorisé avant QUARANTINE_COOLDOWN_S secondes.
    Les ports avec un score >= THREAT_SCORE_PERMANENT sont en quarantine permanente.
    """

    def __init__(self, threat_scores_ref: dict, lock_ref: threading.Lock):
        self._lock         = threading.Lock()
        self._quarantine   = {}     # {(dpid, port): blocked_at_timestamp}
        self._threat_scores = threat_scores_ref
        self._scores_lock   = lock_ref

    def quarantine(self, dpid: int, port: int):
        with self._lock:
            self._quarantine[(dpid, port)] = time.time()
        logger.warning(
            f"[QUARANTINE] dpid={dpid} port={port} mis en quarantine "
            f"— cooldown {QUARANTINE_COOLDOWN_S}s"
        )

    def can_release(self, dpid: int, port: int) -> tuple:
        """
        Retourne (autorisé: bool, raison: str).
        """
        # Vérifier score permanent
        with self._scores_lock:
            score = self._threat_scores.get((dpid, port), 0)
        if score >= THREAT_SCORE_PERMANENT:
            return False, f"Quarantine permanente (threat score={score} >= {THREAT_SCORE_PERMANENT})"

        with self._lock:
            blocked_at = self._quarantine.get((dpid, port))

        if blocked_at is None:
            return True, "Port non en quarantine"

        elapsed   = time.time() - blocked_at
        remaining = QUARANTINE_COOLDOWN_S - elapsed
        if remaining > 0:
            return False, f"Cooldown actif — encore {int(remaining)}s à attendre"

        return True, "Cooldown expiré"

    def release(self, dpid: int, port: int):
        with self._lock:
            self._quarantine.pop((dpid, port), None)
        logger.info(f"[QUARANTINE] dpid={dpid} port={port} libéré")

    def is_quarantined(self, dpid: int, port: int) -> bool:
        with self._lock:
            return (dpid, port) in self._quarantine

    def get_status(self) -> list:
        now = time.time()
        with self._lock:
            result = []
            for (d, p), blocked_at in self._quarantine.items():
                elapsed   = now - blocked_at
                remaining = max(0, QUARANTINE_COOLDOWN_S - elapsed)
                with self._scores_lock:
                    score = self._threat_scores.get((d, p), 0)
                result.append({
                    'dpid':         d,
                    'port':         p,
                    'blocked_since': time.strftime(
                        '%Y-%m-%d %H:%M:%S',
                        time.localtime(blocked_at)
                    ),
                    'cooldown_remaining_s': int(remaining),
                    'permanent': score >= THREAT_SCORE_PERMANENT,
                    'threat_score': score,
                })
            return result


# ── Couche 4 : Threat Scoring ─────────────────────────────────────────────────

class ThreatScorer:
    """
    Chaque événement malveillant incrémente le score d'un port.
    Score >= THREAT_SCORE_BLOCK      → blocage automatique renforcé
    Score >= THREAT_SCORE_PERMANENT  → quarantine permanente
    """

    WEIGHTS = {
        'rogue_dhcp_offer':    2,
        'rogue_dhcp_ack':      3,
        'rogue_dhcp_nak':      1,
        'rate_limit_exceeded': 2,
        'ip_source_violation': 3,
    }

    def __init__(self):
        self._lock   = threading.Lock()
        self._scores = {}   # {(dpid, port): int}
        self._history = []  # log des événements

    def add_event(self, dpid: int, port: int, event_type: str) -> int:
        """Ajoute un événement et retourne le nouveau score."""
        weight = self.WEIGHTS.get(event_type, 1)
        with self._lock:
            self._scores[(dpid, port)] = self._scores.get((dpid, port), 0) + weight
            score = self._scores[(dpid, port)]
            self._history.append({
                'dpid':      dpid,
                'port':      port,
                'event':     event_type,
                'weight':    weight,
                'new_score': score,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            })
            self._history = self._history[-500:]

        level = "CRITICAL" if score >= THREAT_SCORE_PERMANENT else \
                "WARNING"  if score >= THREAT_SCORE_BLOCK else "INFO"
        logger.log(
            logging.CRITICAL if level == "CRITICAL" else
            logging.WARNING  if level == "WARNING"  else logging.INFO,
            f"[THREAT SCORE] dpid={dpid} port={port} "
            f"event={event_type}(+{weight}) → score={score}"
        )
        return score

    def get_score(self, dpid: int, port: int) -> int:
        with self._lock:
            return self._scores.get((dpid, port), 0)

    def reset_score(self, dpid: int, port: int):
        with self._lock:
            self._scores.pop((dpid, port), None)
        logger.info(f"[THREAT SCORE] Score réinitialisé: dpid={dpid} port={port}")

    def get_all_scores(self) -> dict:
        with self._lock:
            return {
                f"dpid={d}/port={p}": s
                for (d, p), s in self._scores.items()
            }

    def get_history(self, limit: int = 50) -> list:
        with self._lock:
            return list(reversed(self._history[-limit:]))

    @property
    def scores_dict(self) -> dict:
        """Référence directe au dict interne (pour QuarantineManager)."""
        return self._scores

    @property
    def lock(self) -> threading.Lock:
        return self._lock


# ── Façade principale ─────────────────────────────────────────────────────────

class DHCPMitigationLayer:
    """
    Point d'entrée unique pour toutes les couches de mitigation.

    Instanciation dans StandaloneSTController.__init__ :
        from dhcp_mitigation import DHCPMitigationLayer
        self.dhcp_mitigation = DHCPMitigationLayer(self)

    Appel dans packet_in_handler, AVANT _check_dhcp_snooping :
        if not self.dhcp_mitigation.inspect(datapath, in_port, dhcp_info, src_ip, src_mac):
            return
    """

    def __init__(self, controller):
        """
        controller : instance de StandaloneSTController
        On récupère les références aux structures existantes (binding, lock).
        """
        self._ctrl = controller

        self.threat_scorer  = ThreatScorer()
        self.rate_limiter   = DHCPRateLimiter()
        self.quarantine_mgr = QuarantineManager(
            threat_scores_ref=self.threat_scorer.scores_dict,
            lock_ref=self.threat_scorer.lock,
        )
        self.ip_source_guard = IPSourceGuard(
            binding_ref=controller.dhcp_snooping.binding,
            lock_ref=controller.dhcp_snooping._lock,
        )

        logger.info("=" * 66)
        logger.info("  [MITIGATION] DHCPMitigationLayer activée")
        logger.info(f"  Rate Limit   : {RATE_LIMIT_MAX_PKTS} paquets / {RATE_LIMIT_WINDOW_S}s")
        logger.info(f"  Quarantine   : cooldown {QUARANTINE_COOLDOWN_S}s")
        logger.info(f"  Threat Score : block>={THREAT_SCORE_BLOCK}  permanent>={THREAT_SCORE_PERMANENT}")
        logger.info("=" * 66)

    # ── Méthode principale ────────────────────────────────────────────────────

    def inspect(
        self,
        datapath,
        in_port: int,
        dhcp_info: Optional[dict],
        src_ip: str = '',
        src_mac: str = '',
    ) -> bool:
        """
        Appelée pour chaque paquet DHCP (dhcp_info != None)
        ET pour chaque paquet IP normal (pour IP Source Guard).

        Retourne True  → paquet autorisé, continuer le traitement normal.
        Retourne False → paquet bloqué par la mitigation, DROP immédiat.
        """
        dpid = datapath.id

        # ── IP Source Guard (tout paquet IP, pas seulement DHCP) ─────────────
        if src_mac and src_ip:
            if not self.ip_source_guard.check(src_mac, src_ip, dpid, in_port):
                score = self.threat_scorer.add_event(dpid, in_port, 'ip_source_violation')
                self._handle_threat(datapath, in_port, score, 'IP_SOURCE_GUARD_VIOLATION',
                                    f"MAC={src_mac} IP={src_ip} incohérent avec binding")
                return False

        if dhcp_info is None:
            return True

        # ── Rate Limiting ─────────────────────────────────────────────────────
        if self.rate_limiter.record_and_check(dpid, in_port):
            score = self.threat_scorer.add_event(dpid, in_port, 'rate_limit_exceeded')
            self._handle_threat(datapath, in_port, score, 'DHCP_RATE_LIMIT',
                                f"Flood DHCP détecté: >{RATE_LIMIT_MAX_PKTS} paquets/{RATE_LIMIT_WINDOW_S}s")
            return False

        # ── Quarantine check ─────────────────────────────────────────────────
        if self.quarantine_mgr.is_quarantined(dpid, in_port):
            logger.warning(f"[MITIGATION] Paquet DHCP rejeté — port en quarantine: dpid={dpid} port={in_port}")
            return False

        # ── Threat scoring pour messages serveur sur ports untrusted ─────────
        if dhcp_info.get('is_server_msg'):
            if not self._ctrl.dhcp_snooping.is_trusted(dpid, in_port):
                msg_type = dhcp_info['msg_type']
                event = {2: 'rogue_dhcp_offer', 5: 'rogue_dhcp_ack', 6: 'rogue_dhcp_nak'}.get(
                    msg_type, 'rogue_dhcp_offer'
                )
                score = self.threat_scorer.add_event(dpid, in_port, event)
                self._handle_threat(datapath, in_port, score, 'ROGUE_DHCP_MSG',
                                    f"MsgType={msg_type} sur port untrusted")
                # On laisse _check_dhcp_snooping existant faire le blocage applicatif
                # On ajoute seulement la quarantine si score élevé
                if score >= THREAT_SCORE_BLOCK:
                    self.quarantine_mgr.quarantine(dpid, in_port)

        return True

    # ── Gestion des menaces ───────────────────────────────────────────────────

    def _handle_threat(self, datapath, port: int, score: int, threat_type: str, detail: str):
        dpid = datapath.id
        logger.critical(
            f"[MITIGATION] {threat_type} dpid={dpid} port={port} "
            f"score={score} — {detail}"
        )
        if score >= THREAT_SCORE_PERMANENT:
            logger.critical(
                f"[MITIGATION] QUARANTINE PERMANENTE dpid={dpid} port={port} "
                f"(score={score} >= {THREAT_SCORE_PERMANENT})"
            )
            self.quarantine_mgr.quarantine(dpid, in_port=port)
            # Renforcer le blocage avec une priorité encore plus haute (200)
            self._do_hard_block(datapath, port)
        elif score >= THREAT_SCORE_BLOCK:
            self.quarantine_mgr.quarantine(dpid, port)

    def _do_hard_block(self, datapath, port: int):
        """
        Installe une règle de blocage priorité 200 (au-dessus du blocage normal p=100).
        Résiste même si la règle p=100 est supprimée par erreur.
        """
        parser  = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match   = parser.OFPMatch(in_port=port)
        mod     = parser.OFPFlowMod(
            datapath=datapath,
            priority=200,
            command=ofproto.OFPFC_ADD,
            match=match,
            instructions=[],   # DROP
        )
        datapath.send_msg(mod)
        logger.critical(
            f"[MITIGATION] Hard-block (priority=200) installé: "
            f"dpid={datapath.id} port={port}"
        )

    # ── API de gestion ────────────────────────────────────────────────────────

    def try_release(self, datapath, port: int) -> tuple:
        """
        Tentative de libération manuelle d'un port.
        Retourne (succès: bool, message: str).
        """
        dpid = datapath.id
        allowed, reason = self.quarantine_mgr.can_release(dpid, port)
        if not allowed:
            return False, reason
        self.quarantine_mgr.release(dpid, port)
        self.rate_limiter.reset(dpid, port)
        return True, f"Port dpid={dpid} port={port} libéré de quarantine"

    def reset_threat_score(self, dpid: int, port: int):
        self.threat_scorer.reset_score(dpid, port)
        self.quarantine_mgr.release(dpid, port)

    def get_full_status(self) -> dict:
        return {
            'config': {
                'rate_limit_max_pkts':    RATE_LIMIT_MAX_PKTS,
                'rate_limit_window_s':    RATE_LIMIT_WINDOW_S,
                'quarantine_cooldown_s':  QUARANTINE_COOLDOWN_S,
                'threat_score_block':     THREAT_SCORE_BLOCK,
                'threat_score_permanent': THREAT_SCORE_PERMANENT,
            },
            'rate_limiter':    self.rate_limiter.get_stats(),
            'quarantine':      self.quarantine_mgr.get_status(),
            'threat_scores':   self.threat_scorer.get_all_scores(),
            'ip_violations':   self.ip_source_guard.get_violations()[:20],
            'threat_history':  self.threat_scorer.get_history(20),
        }
