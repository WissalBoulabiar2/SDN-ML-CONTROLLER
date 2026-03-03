"""
collect_traffic.py — Collecte automatique de trafic Mininet
===========================================================
Lance Mininet, génère du trafic étiqueté (normal + attaques),
collecte les FlowStats via Ryu et exporte un CSV prêt pour l'entraînement.

Usage:
  sudo python experiments/collect_traffic.py

Prérequis:
  - Ryu controller en cours :
      ryu-manager controllers/controller.py --observe-links
  - Mininet + OVS installés
  - hping3, nmap recommandés (optionnels)

Sortie:
  data/mininet_flows.csv   ← dataset d'entraînement
"""

import os
import sys
import time
import logging
import argparse

# Ajouter le répertoire racine au path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description='SDN Traffic Collector')
    p.add_argument('--duration', type=int, default=30,
                   help='Durée par scénario en secondes (défaut: 30)')
    p.add_argument('--switches', type=int, default=4,
                   help='Nombre de switches (défaut: 4)')
    p.add_argument('--hosts', type=int, default=2,
                   help='Hôtes par switch (défaut: 2)')
    p.add_argument('--cooldown', type=int, default=5,
                   help='Pause entre scénarios en secondes (défaut: 5)')
    p.add_argument('--output', type=str,
                   default=os.path.join(ROOT, 'data', 'mininet_flows.csv'),
                   help='Fichier CSV de sortie')
    p.add_argument('--poll', type=int, default=5,
                   help='Intervalle de polling FlowStats en secondes (défaut: 5)')
    p.add_argument('--no-attacks', action='store_true',
                   help='Collecter uniquement du trafic normal')
    return p.parse_args()


def check_prerequisites():
    """Vérifie que les outils nécessaires sont disponibles."""
    errors = []

    # Vérifier les imports Mininet
    try:
        from mininet.net import Mininet
        from mininet.node import RemoteController
    except ImportError:
        errors.append("mininet non installé → sudo apt install mininet")

    # Vérifier OVS
    ret = os.system('which ovs-ofctl > /dev/null 2>&1')
    if ret != 0:
        errors.append("ovs-ofctl absent → sudo apt install openvswitch-switch")

    # Vérifier qu'on est root
    if os.geteuid() != 0:
        errors.append("doit être lancé en root → sudo python experiments/collect_traffic.py")

    if errors:
        for e in errors:
            logger.error(f"✗ {e}")
        sys.exit(1)

    # Vérifier outils optionnels
    for tool in ['hping3', 'nmap']:
        ret = os.system(f'which {tool} > /dev/null 2>&1')
        if ret != 0:
            logger.warning(f"⚠ {tool} absent (optionnel) — fallback activé")


def wait_for_controller(timeout: int = 30):
    """Attend que le contrôleur Ryu soit prêt."""
    import socket
    logger.info("Attente du contrôleur Ryu (port 6633)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            s = socket.create_connection(('127.0.0.1', 6633), timeout=1)
            s.close()
            logger.info("✓ Contrôleur Ryu connecté")
            return True
        except (ConnectionRefusedError, OSError):
            time.sleep(1)
    logger.error("✗ Contrôleur Ryu non disponible sur port 6633")
    logger.error("  Lancer d'abord: ryu-manager controllers/controller.py --observe-links")
    return False


def main():
    args = parse_args()

    print("""
╔══════════════════════════════════════════════════════╗
║      SDN Traffic Collector — Ryu + Mininet          ║
║   Génère un dataset étiqueté depuis votre réseau    ║
╚══════════════════════════════════════════════════════╝
""")

    # ── Vérifications ────────────────────────────────────────────────────────
    check_prerequisites()

    if not wait_for_controller():
        sys.exit(1)

    # ── Imports Mininet ──────────────────────────────────────────────────────
    from mininet.net import Mininet
    from mininet.node import RemoteController, OVSSwitch
    from mininet.link import TCLink
    from mininet.topo import Topo
    from mininet.log import setLogLevel
    from mininet.clean import cleanup

    from utils.mininet_helper import TrafficGenerator

    setLogLevel('warning')

    # ── Nettoyage préalable ──────────────────────────────────────────────────
    logger.info("Nettoyage Mininet préalable...")
    cleanup()
    time.sleep(1)

    # ── Topologie ────────────────────────────────────────────────────────────
    class CollectTopo(Topo):
        def build(self, n_sw=4, h_per_sw=2):
            link_opts = dict(bw=10, delay='5ms', use_htb=True)
            switches = [self.addSwitch(f's{i}',
                                       protocols='OpenFlow13')
                        for i in range(1, n_sw + 1)]
            hcount = 0
            for sw in switches:
                for _ in range(h_per_sw):
                    hcount += 1
                    host = self.addHost(
                        f'h{hcount}',
                        ip=f'10.0.0.{hcount}/24',
                        defaultRoute=f'via 10.0.0.1'
                    )
                    self.addLink(host, sw, **link_opts)
            # Lien linéaire entre switches
            for i in range(len(switches) - 1):
                self.addLink(switches[i], switches[i + 1], **link_opts)

    topo = CollectTopo(n_sw=args.switches, h_per_sw=args.hosts)

    # ── Démarrage Mininet ────────────────────────────────────────────────────
    logger.info(f"Démarrage Mininet: {args.switches} switches, "
                f"{args.switches * args.hosts} hôtes...")

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name, ip='127.0.0.1', port=6633),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=True,
        waitConnected=True,
    )

    net.start()
    time.sleep(3)  # Laisser le temps au contrôleur de découvrir la topo

    # Afficher la topologie
    logger.info("Topologie démarrée:")
    for sw in net.switches:
        logger.info(f"  Switch {sw.name} connecté")
    for h in net.hosts:
        logger.info(f"  Hôte {h.name} IP={h.IP()}")

    # Test de connectivité initial
    logger.info("Test de connectivité (pingAll)...")
    net.pingAll(timeout=2)

    # ── Collecte ─────────────────────────────────────────────────────────────
    gen = TrafficGenerator(net, poll_interval=args.poll)

    try:
        if args.no_attacks:
            # Mode normal uniquement
            from utils.mininet_helper import NormalTraffic
            logger.info("Mode: trafic normal uniquement")
            gen.run_scenario(NormalTraffic, duration=args.duration * 3)
        else:
            # Tous les scénarios
            gen.run_all_scenarios(
                duration_each=args.duration,
                cooldown=args.cooldown
            )

    except KeyboardInterrupt:
        logger.warning("\n⚠ Collecte interrompue par l'utilisateur")

    finally:
        # ── Export CSV ───────────────────────────────────────────────────────
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        csv_path = gen.export_csv(path=args.output)

        # ── Arrêt Mininet ────────────────────────────────────────────────────
        logger.info("Arrêt de Mininet...")
        net.stop()
        cleanup()

        # ── Résumé ───────────────────────────────────────────────────────────
        n = len(gen._collected)
        print(f"""
╔══════════════════════════════════════════════════════╗
║                   COLLECTE TERMINÉE                 ║
╠══════════════════════════════════════════════════════╣
║  Flows collectés : {n:<33d}║
║  Fichier CSV     : {os.path.basename(csv_path):<33s}║
║                                                      ║
║  Prochaine étape :                                  ║
║    python experiments/train_ids.py                  ║
╚══════════════════════════════════════════════════════╝
""")

        if n < 100:
            logger.warning(
                f"⚠ Seulement {n} flows collectés. "
                f"Augmentez --duration ou vérifiez la connectivité Mininet."
            )

        return csv_path


if __name__ == '__main__':
    main()

