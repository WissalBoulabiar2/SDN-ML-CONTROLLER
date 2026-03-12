"""
stp_exporter.py — Prometheus exporter pour controller_stp.py
=============================================================
Scrape les endpoints REST du contrôleur toutes les 10s
et expose les métriques sur http://localhost:9101/metrics

Installation :
    pip install prometheus-client requests

Lancement :
    python3 stp_exporter.py

Endpoints scrapés :
    GET /stp/status
    GET /stp/pending
    GET /stp/security/alerts
    GET /dhcp-snooping/status
    GET /dhcp-mitigation/status
    GET /dhcp-mitigation/threats
"""

import time
import logging
import requests
from prometheus_client import start_http_server, Gauge, Counter, Info

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONTROLLER_URL = "http://localhost:8080"   # adresse de ton contrôleur Ryu
EXPORTER_PORT  = 9101                      # port exposé à Prometheus
SCRAPE_INTERVAL = 10                       # secondes entre chaque scrape

# ── Métriques Prometheus ──────────────────────────────────────────────────────

# STP
stp_ports_total = Gauge(
    'stp_ports_total', 'Nombre total de ports STP trackés')
stp_ports_blocking = Gauge(
    'stp_ports_blocking', 'Ports en état BLOCKING/DISCARDING')
stp_ports_forwarding = Gauge(
    'stp_ports_forwarding', 'Ports en état FORWARDING')
stp_ports_learning = Gauge(
    'stp_ports_learning', 'Ports en état LEARNING')
stp_pending_actions = Gauge(
    'stp_pending_actions', 'Actions en attente de confirmation manuelle')
stp_protocol_count = Gauge(
    'stp_protocol_count', 'Paquets détectés par protocole STP',
    ['protocol'])

# STP Security
stp_security_alerts_total = Gauge(
    'stp_security_alerts_total', 'Nombre total d alertes sécurité STP')
stp_protected_ports = Gauge(
    'stp_protected_ports', 'Ports avec BPDU Guard ou Root Guard actif')

# DHCP Snooping
dhcp_trusted_ports = Gauge(
    'dhcp_trusted_ports', 'Ports DHCP trusted')
dhcp_blocked_rogue_ports = Gauge(
    'dhcp_blocked_rogue_ports', 'Ports bloqués pour DHCP rogue')
dhcp_binding_entries = Gauge(
    'dhcp_binding_entries', 'Entrées dans la binding table MAC→IP')
dhcp_rogue_alerts_total = Gauge(
    'dhcp_rogue_alerts_total', 'Nombre total d alertes rogue DHCP')

# DHCP Mitigation
mitigation_quarantined_ports = Gauge(
    'mitigation_quarantined_ports', 'Ports actuellement en quarantine')
mitigation_permanent_quarantine = Gauge(
    'mitigation_permanent_quarantine', 'Ports en quarantine permanente')
mitigation_threat_score = Gauge(
    'mitigation_threat_score', 'Threat score par port',
    ['dpid_port'])
mitigation_ip_violations_total = Gauge(
    'mitigation_ip_violations_total', 'Violations IP Source Guard détectées')
mitigation_rate_limit_hits = Gauge(
    'mitigation_rate_limit_pkts', 'Paquets DHCP en cours dans la fenêtre rate-limit',
    ['dpid_port'])

# Santé de l'exporteur
exporter_scrape_errors = Counter(
    'stp_exporter_scrape_errors_total', 'Erreurs de scrape du contrôleur')
exporter_last_scrape = Gauge(
    'stp_exporter_last_scrape_timestamp', 'Timestamp du dernier scrape réussi')


# ── Fonctions de scrape ───────────────────────────────────────────────────────

def fetch(path):
    """Appelle l'API REST et retourne le JSON, ou None en cas d'erreur."""
    try:
        r = requests.get(f"{CONTROLLER_URL}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"[SCRAPE] Erreur {path}: {e}")
        exporter_scrape_errors.inc()
        return None


def scrape_stp_status():
    data = fetch("/stp/status")
    if not data:
        return

    port_states = data.get('port_states', {})
    stp_ports_total.set(len(port_states))

    blocking = forwarding = learning = 0
    for info in port_states.values():
        state = info.get('state', '')
        if state in ('blocking', 'discarding', 'root_inconsistent',
                     'bpdu_guard_err', 'dhcp_rogue_err'):
            blocking += 1
        elif state == 'forwarding':
            forwarding += 1
        elif state == 'learning':
            learning += 1

    stp_ports_blocking.set(blocking)
    stp_ports_forwarding.set(forwarding)
    stp_ports_learning.set(learning)

    for proto, count in data.get('protocols_detected', {}).items():
        stp_protocol_count.labels(protocol=proto).set(count)

    security = data.get('stp_security', {})
    stp_security_alerts_total.set(security.get('alert_count', 0))
    stp_protected_ports.set(len(security.get('protected_ports', {})))

    dhcp = data.get('dhcp_snooping', {})
    dhcp_trusted_ports.set(len(dhcp.get('trusted_ports', [])))
    dhcp_blocked_rogue_ports.set(len(dhcp.get('blocked_rogue_ports', [])))
    dhcp_binding_entries.set(len(dhcp.get('binding_table', {})))
    dhcp_rogue_alerts_total.set(dhcp.get('rogue_alerts', 0))


def scrape_pending():
    data = fetch("/stp/pending")
    if not data:
        return
    stp_pending_actions.set(data.get('count', 0))


def scrape_mitigation():
    data = fetch("/dhcp-mitigation/status")
    if not data:
        return

    quarantine_list = data.get('quarantine', [])
    mitigation_quarantined_ports.set(len(quarantine_list))
    mitigation_permanent_quarantine.set(
        sum(1 for q in quarantine_list if q.get('permanent', False))
    )

    mitigation_ip_violations_total.set(
        len(data.get('ip_violations', []))
    )

    # Threat scores par port
    for dpid_port, score in data.get('threat_scores', {}).items():
        mitigation_threat_score.labels(dpid_port=dpid_port).set(score)

    # Rate limiter
    for dpid_port, count in data.get('rate_limiter', {}).items():
        mitigation_rate_limit_hits.labels(dpid_port=dpid_port).set(count)


# ── Boucle principale ─────────────────────────────────────────────────────────

def collect():
    scrape_stp_status()
    scrape_pending()
    scrape_mitigation()
    exporter_last_scrape.set(time.time())
    logger.info(f"[SCRAPE] OK — {time.strftime('%H:%M:%S')}")


if __name__ == '__main__':
    logger.info(f"[EXPORTER] Démarrage sur port {EXPORTER_PORT}")
    logger.info(f"[EXPORTER] Contrôleur cible : {CONTROLLER_URL}")
    start_http_server(EXPORTER_PORT)
    logger.info(f"[EXPORTER] Métriques disponibles sur http://localhost:{EXPORTER_PORT}/metrics")

    while True:
        try:
            collect()
        except Exception as e:
            logger.error(f"[EXPORTER] Erreur collect: {e}")
            exporter_scrape_errors.inc()
        time.sleep(SCRAPE_INTERVAL)
