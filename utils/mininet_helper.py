"""
Mininet Traffic Generator + Feature Collector (FIXED VERSION)
===========================================================
"""

import time
import logging
import os
import threading
import csv
import random
from typing import List, Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Helpers réseau ──────────────────────────────────────────────────────────

def cmd_bg(host, cmd: str):
    host.cmd(cmd + ' &')

def cmd_wait(host, cmd: str, timeout: int = 30) -> str:
    return host.cmd(f'timeout {timeout} {cmd} 2>/dev/null || true')

def iperf_server(host):
    host.cmd('pkill iperf 2>/dev/null; iperf -s -D')
    time.sleep(0.5)

def stop_all(net):
    for h in net.hosts:
        h.cmd('pkill hping3 2>/dev/null || true')
        h.cmd('pkill iperf  2>/dev/null || true')
        h.cmd('pkill nmap   2>/dev/null || true')
        h.cmd('pkill ping   2>/dev/null || true')

def check_tool(host, tool: str) -> bool:
    result = host.cmd(f'which {tool} 2>/dev/null')
    return bool(result.strip())

# ── Scénarios de trafic (Classes simplifiées pour l'exemple) ────────────────

class TrafficScenario:
    name, label, attack_cat = 'base', 0, 'Normal'
    def run(self, net, duration: int): raise NotImplementedError
    def stop(self, net): pass

class NormalTraffic(TrafficScenario):
    name, label, attack_cat = 'Normal', 0, 'Normal'
    def run(self, net, duration: int):
        hosts = net.hosts
        for i in range(0, len(hosts), 2): iperf_server(hosts[i])
        end = time.time() + duration
        while time.time() < end:
            src, dst = random.sample(hosts, 2)
            cmd_bg(src, f'ping -c 2 {dst.IP()}')
            time.sleep(random.uniform(0.5, 1.5))

class DoSAttack(TrafficScenario):
    name, label, attack_cat = 'DoS', 1, 'DoS'
    def run(self, net, duration: int):
        atk, victim = net.hosts[0], net.hosts[-1]
        logger.warning(f"[DoS] SYN flood: {atk.name} -> {victim.name}")
        if check_tool(atk, 'hping3'):
            cmd_bg(atk, f'hping3 -S --flood --rand-source {victim.IP()}')
        else:
            cmd_bg(atk, f'ping -f {victim.IP()}')
        time.sleep(duration)
    def stop(self, net): stop_all(net)

# ... (Les autres scénarios DDoS, Probe restent identiques à ton code précédent)

# ── Gestionnaire de collecte ────────────────────────────────────────────────

class TrafficGenerator:
    SCENARIOS = [NormalTraffic, DoSAttack] # Ajoute les autres ici

    def __init__(self, net, poll_interval: int = 5):
        self.net = net
        self.poll_interval = poll_interval
        self._collected = []
        self._current_label = 0
        self._current_attack_cat = 'Normal'
        self._collecting = False

    def run_scenario(self, scenario_cls, duration: int = 30):
        scenario = scenario_cls()
        self._current_label = scenario.label
        self._current_attack_cat = scenario.name
        self._collecting = True
        collect_thread = threading.Thread(target=self._collect_loop, daemon=True)
        collect_thread.start()
        try:
            scenario.run(self.net, duration)
        finally:
            scenario.stop(self.net)
            self._collecting = False
            collect_thread.join(timeout=2)

    def run_all_scenarios(self, duration_each=30, cooldown=5):
        for i, cls in enumerate(self.SCENARIOS, 1):
            self.run_scenario(cls, duration=duration_each)
            time.sleep(cooldown)

    def _collect_loop(self):
        from utils.feature_extraction import FeatureExtractor
        extractor = FeatureExtractor()
        while self._collecting:
            try:
                flows = self._poll_flows_ovs(extractor)
                for f in flows:
                    f['label'] = self._current_label
                    f['attack_cat'] = self._current_attack_cat
                    self._collected.append(f)
            except Exception as e:
                logger.debug(f"Collect error: {e}")
            time.sleep(self.poll_interval)

    def _poll_flows_ovs(self, extractor) -> List[dict]:
        results = []
        for sw in self.net.switches:
            # FIX: Ajout de -O OpenFlow13
            raw = sw.cmd(f'ovs-ofctl -O OpenFlow13 dump-flows {sw.name} 2>/dev/null')
            parsed = _parse_ovs_flows(raw, sw.name)
            for p in parsed:
                feat = _ovs_flow_to_features(p, extractor)
                if feat: results.append(feat)
        return results

    def export_csv(self, path=None):
        from utils.feature_extraction import UNSW_COLUMNS
        path = path or os.path.join(DATA_DIR, 'mininet_flows.csv')
        if not self._collected: return path
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=UNSW_COLUMNS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(self._collected)
        return path

# ── Parser OVS (CORRIGÉ) ────────────────────────────────────────────────────

def _parse_ovs_flows(raw_output: str, sw_name: str) -> List[dict]:
    flows = []
    for line in raw_output.splitlines():
        line = line.strip()
        # On ignore les lignes vides ou les flux par défaut (priority 0)
        if 'cookie=' not in line or 'priority=0' in line: 
            continue
        
        try:
            f = {'switch': sw_name}
            # Extraction des stats de base
            f['duration'] = float(line.split('duration=')[1].split('s')[0])
            f['n_packets'] = int(line.split('n_packets=')[1].split(',')[0])
            f['n_bytes'] = int(line.split('n_bytes=')[1].split(',')[0])

            # --- CORRECTION MAJEURE ICI ---
            # On isole la partie "match" qui se trouve entre priority et actions
            parts = line.split('priority=')[1].split(',')
            # On cherche les adresses dans ces parties
            f['src_ip'] = ''
            f['dst_ip'] = ''
            
            for p in parts:
                if 'nw_src=' in p: f['src_ip'] = p.split('=')[1]
                elif 'nw_dst=' in p: f['dst_ip'] = p.split('=')[1]
                elif 'dl_src=' in p: f['src_ip'] = p.split('=')[1] # Fallback MAC
                elif 'dl_dst=' in p: f['dst_ip'] = p.split('=')[1] # Fallback MAC

            # Si on n'a toujours pas d'adresse, on utilise un placeholder pour ne pas rejeter le flux
            if not f['src_ip']: f['src_ip'] = "00:00:00:00:00:00"
            if not f['dst_ip']: f['dst_ip'] = "00:00:00:00:00:00"

            # Protocole
            f['proto'] = 'tcp' if 'tcp' in line else 'udp' if 'udp' in line else 'icmp' if 'icmp' in line else 'other'
            f['src_port'] = int(line.split('tp_src=')[1].split(',')[0]) if 'tp_src=' in line else 0
            f['dst_port'] = int(line.split('tp_dst=')[1].split(',')[0]) if 'tp_dst=' in line else 0
            
            flows.append(f)
        except Exception:
            continue
    return flows

def _ovs_flow_to_features(flow: dict, extractor) -> Optional[dict]:
    # Cette fonction transforme les données brutes en format UNSW-NB15
    # Elle utilise ton FeatureExtractor existant.
    try:
        dur = max(flow['duration'], 0.001)
        return extractor._build_feat(
            src_ip=flow['src_ip'], dst_ip=flow['dst_ip'],
            sp=flow['src_port'], dp=flow['dst_port'],
            proto=flow['proto'], service='-', state='CON',
            dur=dur, spkts=flow['n_packets'], dpkts=0,
            sbytes=flow['n_bytes'], dbytes=0,
            sttl=64, sload=(flow['n_bytes']*8)/dur,
            smean=flow['n_bytes']//max(flow['n_packets'],1),
            swin=255, dpid=0, sinpkt=dur/max(flow['n_packets'],1),
            rate=flow['n_packets']/dur
        )
    except: return None
