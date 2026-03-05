"""
SDN Routing: Q-Learning (congestion-aware) vs Dijkstra
=======================================================
Différence clé avec la v1 :
  - Des nœuds "bottleneck" sont injectés (mu * 0.3)
  - L'état RL inclut le taux d'utilisation local (load_bucket)
  - La reward pénalise fortement les nœuds saturés
  - => Le RL apprend à CONTOURNER la congestion, Dijkstra ne le fait pas

4 figures : métriques par topologie, convergence reward,
            topologie avec bottlenecks colorés, bar chart résumé final.
"""

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import pickle
import os
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# 1.  MODÈLE RÉSEAU  M/M/1/K  avec bottlenecks
# ══════════════════════════════════════════════════════════════════════════════

class SDNNetwork:
    """
    Réseau SDN avec files M/M/1/K.
    Certains nœuds sont des bottlenecks (mu * 0.3) :
      - Dijkstra les ignore  (aveugle à la congestion)
      - RL les détecte et les contourne
    """

    def __init__(self, G: nx.Graph, mu: float = 600, K: int = 20, seed: int = 42):
        self.G  = G
        rng = random.Random(seed)
        self.mu = {n: mu for n in G.nodes()}
        self.K  = {n: K  for n in G.nodes()}
        # Injecter des bottlenecks sur ~25 % des nœuds
        nodes = list(G.nodes())
        bn = rng.sample(nodes, max(1, len(nodes) // 4)) if len(nodes) >= 4 else []
        for n in bn:
            self.mu[n] = mu * 0.3
        self._bottlenecks = bn
    def add_node(self, n, mu=600, K=20):
        if n not in self.mu:
            self.mu[n] = mu
            self.K[n]  = K
    def bottleneck_nodes(self):
        return self._bottlenecks

    # ── M/M/1/K ──────────────────────────────────────────────────────────────

    def _rho(self, lam, n):
        return min(lam / self.mu[n], 0.9999) if self.mu[n] > 0 else 0

    def _pb(self, lam, n):
        r = self._rho(lam, n); K = self.K[n]
        if r == 0: return 0.0
        if abs(r - 1) < 1e-9: return 1 / (K + 1)
        return ((1 - r) * r**K) / (1 - r**(K + 1))

    def _en(self, lam, n):
        r = self._rho(lam, n); K = self.K[n]
        if r == 0: return 0.0
        if abs(r - 1) < 1e-9: return K / 2.0
        return (r / (1 - r)) - ((K + 1) * r**(K + 1)) / (1 - r**(K + 1))

    def node_delay(self, lam, n):
        pb = self._pb(lam, n); en = self._en(lam, n)
        eff = lam * (1 - pb)
        return en / eff if eff > 1e-9 else 50.0   # valeur haute si saturé

    def node_cost(self, lam, n):
        """Score de coût pour le RL : pénalise fortement les bottlenecks."""
        return self.node_delay(lam, n) * 100 + self._pb(lam, n) * 50

    def path_metrics(self, path, lam):
        """Retourne (delay_total, loss_total) sur un chemin."""
        load = lam; total_d = total_l = 0.0
        for n in path:
            pb       = self._pb(load, n)
            total_d += self.node_delay(load, n)
            total_l += load * pb
            load    *= (1 - pb)
        return total_d, total_l

    def evaluate(self, routing_paths: dict, flow_rates: dict):
        D, L, S = [], [], []
        for fd, path in routing_paths.items():
            if path is None or len(path) < 2: continue
            lam = flow_rates.get(fd, 100)
            d, l = self.path_metrics(path, lam)
            D.append(d); L.append(l); S.append(lam)
        avg_d = float(np.mean(D)) if D else 0.0
        tot_l = float(np.sum(L))  if L else 0.0
        tp    = float(np.sum(S))  - tot_l
        return avg_d, tot_l, tp


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DIJKSTRA BASELINE  (hop-count, aveugle à la congestion)
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra_routing(G: nx.Graph, flows: list) -> dict:
    """Chemin le plus court en nombre de sauts. Ne voit pas les bottlenecks."""
    paths = {}
    for s, d in flows:
        try:    paths[(s, d)] = nx.shortest_path(G, s, d)
        except: paths[(s, d)] = None
    return paths


# ══════════════════════════════════════════════════════════════════════════════
# 3.  AGENT RL  –  Q-Learning congestion-aware (per-hop)
# ══════════════════════════════════════════════════════════════════════════════

class QRoutingAgent:
    """
    Q-Learning hop-by-hop conscient de la congestion.

    État   : (nœud_courant, destination, load_bucket)
             load_bucket = floor(rho / 0.2)  ∈ {0,1,2,3,4}
    Action : prochain voisin
    Reward : -node_cost(suivant)  + bonus si destination atteinte

    L'agent apprend à éviter les nœuds bottleneck
    que Dijkstra prend sans hésiter.
    """

    def __init__(self, G: nx.Graph, net: SDNNetwork,
                 alpha=0.2, gamma=0.9,
                 eps0=1.0, epsf=0.02, epsd=0.9985):
        self.G     = G
        self.net   = net
        self.alpha = alpha
        self.gamma = gamma
        self.eps   = eps0
        self.epsf  = epsf
        self.epsd  = epsd
        self.Q: dict = defaultdict(lambda: defaultdict(float))

    def _load_bucket(self, lam, n):
        return min(int(self.net._rho(lam, n) / 0.2), 4)

    def _state(self, node, dst, lam):
        return (node, dst, self._load_bucket(lam, node))

    def choose(self, node, dst, lam, train=True):
        nb = list(self.G.neighbors(node))
        if not nb: return None
        if train and random.random() < self.eps:
            return random.choice(nb)
        s = self._state(node, dst, lam)
        return max(nb, key=lambda x: self.Q[s][x])

    def update(self, node, dst, lam, action, reward, next_node):
        s      = self._state(node, dst, lam)
        lam_n  = lam * (1 - self.net._pb(lam, node))
        s_n    = self._state(next_node, dst, lam_n)
        nb_n   = list(self.G.neighbors(next_node))
        max_q  = (max(self.Q[s_n][x] for x in nb_n)
                  if nb_n and next_node != dst else 0.0)
        old = self.Q[s][action]
        self.Q[s][action] = old + self.alpha * (reward + self.gamma * max_q - old)

    def get_path(self, src, dst, lam, fallback=None):
        path = [src]; node = src; vis = {src}; load = lam
        for _ in range(len(self.G.nodes) * 3):
            if node == dst: break
            nb = [x for x in self.G.neighbors(node) if x not in vis]
            if not nb: return fallback
            s   = self._state(node, dst, load)
            nxt = max(nb, key=lambda x: self.Q[s][x])
            path.append(nxt); vis.add(nxt)
            load *= (1 - self.net._pb(load, node))
            node  = nxt
        return path if path[-1] == dst else fallback

    def train_episode(self, flows, rates):
        total_r = 0.0
        random.shuffle(flows)
        for src, dst in flows:
            lam  = rates[(src, dst)]
            node = src; vis = {src}; load = lam
            for _ in range(len(self.G.nodes) * 2):
                if node == dst: break
                a = self.choose(node, dst, load, train=True)
                if a is None: break
                cost   = self.net.node_cost(load, a)
                reward = -cost + (200.0 if a == dst else 0.0)
                self.update(node, dst, load, a, reward, a)
                total_r += reward
                if a in vis: break
                vis.add(a); load *= (1 - self.net._pb(load, node)); node = a
        self.eps = max(self.epsf, self.eps * self.epsd)
        return total_r / max(len(flows), 1)

    def save(self, path: str = "qtable.pkl"):
        with open(path, "wb") as f: pickle.dump(dict(self.Q), f)
        print(f"[RL] Q-table saved → {path}")

    def load(self, path: str = "qtable.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.Q = defaultdict(lambda: defaultdict(float), pickle.load(f))
            print(f"[RL] Q-table loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TOPOLOGY FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_topology(topo: str = "random", **kw) -> nx.Graph:
    if topo == "grid":
        return nx.convert_node_labels_to_integers(
            nx.grid_2d_graph(kw.get("n", 5), kw.get("n", 5)))
    if topo == "geant":
        return nx.barabasi_albert_graph(24, 3, seed=42)
    if topo == "custom":
        G = nx.Graph(); G.add_edges_from(kw.get("edges", [])); return G
    n = kw.get("n", 15); p = kw.get("p", 0.3)
    G = nx.erdos_renyi_graph(n, p, seed=42)
    t = 0
    while not nx.is_connected(G) and t < 100:
        G = nx.erdos_renyi_graph(n, p, seed=random.randint(0, 9999)); t += 1
    return G


# ══════════════════════════════════════════════════════════════════════════════
# 5.  EXPÉRIENCE
# ══════════════════════════════════════════════════════════════════════════════

def run_experiment(topo_name: str = "random",
                   n_flows: int = 25,
                   n_episodes: int = 4000,
                   lam_range: tuple = (100, 600)):
    print(f"\n{'═'*50}")
    print(f"  Topologie : {topo_name.upper()}")
    print(f"{'═'*50}")

    if topo_name == "grid":   G = build_topology("grid", n=5)
    elif topo_name == "geant": G = build_topology("geant")
    else:                      G = build_topology("random", n=15, p=0.3)

    net   = SDNNetwork(G, mu=600, K=20, seed=42)
    nodes = list(G.nodes())
    pairs = [(s, d) for s in nodes for d in nodes
             if s != d and nx.has_path(G, s, d)]
    flows      = random.sample(pairs, min(n_flows, len(pairs)))
    flow_rates = {fd: random.uniform(*lam_range) for fd in flows}

    print(f"  Bottleneck nodes : {net.bottleneck_nodes()}")

    # Dijkstra
    dijk_paths = dijkstra_routing(G, flows)
    d0, l0, t0 = net.evaluate(dijk_paths, flow_rates)
    print(f"  [Dijkstra]  Delay={d0:.4f}s  Loss={l0:.1f}  TP={t0:.1f}")

    # RL
    agent = QRoutingAgent(G, net)
    rl_delays, rl_losses, rl_tps, rh = [], [], [], []
    ev = max(1, n_episodes // 100)

    for ep in range(n_episodes):
        rh.append(agent.train_episode(flows, flow_rates))
        if ep % ev == 0:
            rp = {fd: agent.get_path(fd[0], fd[1], flow_rates[fd],
                                     fallback=dijk_paths[fd])
                  for fd in flows}
            d, l, t = net.evaluate(rp, flow_rates)
            rl_delays.append(d); rl_losses.append(l); rl_tps.append(t)

    print(f"  [RL final]  Delay={rl_delays[-1]:.4f}s  Loss={rl_losses[-1]:.1f}  TP={rl_tps[-1]:.1f}")
    pct = (d0 - rl_delays[-1]) / d0 * 100 if d0 > 0 else 0
    print(f"  Amélioration délai vs Dijkstra : {pct:+.1f}%")

    agent.save(f"qtable_{topo_name}.pkl")

    return {
        "topo": topo_name, "G": G, "net": net,
        "dijk": (d0, l0, t0),
        "rl_delays": rl_delays, "rl_losses": rl_losses, "rl_tps": rl_tps,
        "reward_hist": rh, "n_episodes": n_episodes, "eval_every": ev,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURES  — basées sur les résultats réels après entraînement
# ══════════════════════════════════════════════════════════════════════════════

def plot_all(results: list, out_dir: str = "."):
    n      = len(results)
    topos  = [r["topo"].upper() for r in results]
    x_     = np.arange(len(topos))
    w_     = 0.32

    # Extraire les valeurs finales réelles
    dijk_delay = [r["dijk"][0]          for r in results]
    dijk_loss  = [r["dijk"][1]          for r in results]
    dijk_tp    = [r["dijk"][2]          for r in results]
    rl_delay   = [r["rl_delays"][-1]    for r in results]
    rl_loss    = [r["rl_losses"][-1]    for r in results]
    rl_tp      = [r["rl_tps"][-1]       for r in results]

    # ── Figure 1 : Bar chart — 3 métriques × 3 topologies ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    specs = [
        (axes[0], dijk_delay, rl_delay, "Avg End-to-End Delay (s)",  "End-to-End Delay"),
        (axes[1], dijk_loss,  rl_loss,  "Packet Loss (pkt/s)",        "Packet Loss"),
        (axes[2], dijk_tp,    rl_tp,    "Throughput (pkt/s)",         "Throughput"),
    ]
    for ax, dv, rv, yl, ti in specs:
        bd = ax.bar(x_ - w_/2, dv, w_, label='Dijkstra',      color='#27ae60', alpha=0.88, zorder=3)
        br = ax.bar(x_ + w_/2, rv, w_, label='RL (Q-Learning)', color='#e74c3c', alpha=0.88, zorder=3)
        for bar in bd:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                    f"{h:.4f}" if "Delay" in ti else f"{h:.0f}",
                    ha='center', va='bottom', fontsize=8, color='#1a5c35', fontweight='bold')
        for bar in br:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h * 1.01,
                    f"{h:.4f}" if "Delay" in ti else f"{h:.0f}",
                    ha='center', va='bottom', fontsize=8, color='#8b0000', fontweight='bold')
        ax.set_xticks(x_); ax.set_xticklabels(topos, fontsize=11)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(ti, fontsize=11, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3, axis='y', zorder=0)
        ax.set_ylim(0, max(max(dv), max(rv)) * 1.2)
    plt.suptitle("Final Results: RL vs Dijkstra — SDN Routing Optimization",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "fig1_bar_metrics.png")
    plt.savefig(p1, dpi=160, bbox_inches='tight'); plt.close()
    print(f"[✓] {p1}")

    # ── Figure 2 : % amélioration RL vs Dijkstra ─────────────────────────────
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    pct_delay = [(d - r) / d * 100 for d, r in zip(dijk_delay, rl_delay)]
    pct_loss  = [(d - r) / d * 100 for d, r in zip(dijk_loss,  rl_loss)]
    pct_tp    = [(r - d) / d * 100 for d, r in zip(dijk_tp,    rl_tp)]
    specs2 = [
        (axes2[0], pct_delay, "Delay improvement (%)"),
        (axes2[1], pct_loss,  "Loss improvement (%)"),
        (axes2[2], pct_tp,    "Throughput gain (%)"),
    ]
    for ax, pcts, yl in specs2:
        bar_colors = ['#e74c3c' if p >= 0 else '#95a5a6' for p in pcts]
        bars = ax.bar(topos, pcts, color=bar_colors, alpha=0.88, width=0.45, zorder=3)
        for bar, pct in zip(bars, pcts):
            h = bar.get_height()
            offset = 0.3 if h >= 0 else -1.5
            ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                    f"{pct:+.1f}%", ha='center', va='bottom',
                    fontsize=12, fontweight='bold',
                    color='#c0392b' if pct >= 0 else '#7f8c8d')
        ax.axhline(0, color='black', lw=1.2)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(yl, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y', zorder=0)
        margin = max(abs(p) for p in pcts) * 0.3
        ax.set_ylim(min(0, min(pcts)) - margin, max(0, max(pcts)) + margin + 2)
        ax.tick_params(axis='x', labelsize=11)
    plt.suptitle("RL Improvement over Dijkstra (%) — per Topology & Metric",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "fig2_improvement_pct.png")
    plt.savefig(p2, dpi=160, bbox_inches='tight'); plt.close()
    print(f"[✓] {p2}")

    # ── Figure 3 : Convergence reward ────────────────────────────────────────
    fig3, ax3s = plt.subplots(1, n, figsize=(6 * n, 4))
    if n == 1: ax3s = [ax3s]
    for ax, res in zip(ax3s, results):
        rh = np.array(res["reward_hist"])
        w  = max(1, len(rh) // 60)
        ma = np.convolve(rh, np.ones(w) / w, mode='valid')
        ax.plot(rh, color='#ccc', lw=0.5, alpha=0.4, label='Raw')
        ax.plot(np.arange(len(ma)), ma, color='#e74c3c', lw=2.2,
                label=f'Moving avg (w={w})')
        ax.set_title(f"{res['topo'].upper()} – Reward Convergence")
        ax.set_xlabel("Episodes"); ax.set_ylabel("Avg Reward")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.suptitle("RL Training Convergence", fontsize=13, fontweight='bold')
    plt.tight_layout()
    p3 = os.path.join(out_dir, "fig3_convergence.png")
    plt.savefig(p3, dpi=150, bbox_inches='tight'); plt.close()
    print(f"[✓] {p3}")

    # ── Figure 4 : Topologie avec bottlenecks colorés ────────────────────────
    last = results[-1]; G = last["G"]; net = last["net"]
    bn   = net.bottleneck_nodes()
    clrs = ['#e74c3c' if nd in bn else '#2980b9' for nd in G.nodes()]
    plt.figure(figsize=(8, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color=clrs,
            node_size=600, font_color='white', font_size=9,
            edge_color='#95a5a6', width=1.8)
    plt.legend(
        handles=[Patch(color='#e74c3c', label='Bottleneck (mu×0.3)'),
                 Patch(color='#2980b9', label='Normal switch')],
        loc='upper left', fontsize=9)
    plt.title(f"SDN Topology – {last['topo'].upper()}"
              f"  ({len(G.nodes())} nodes, {len(G.edges())} links)", fontsize=12)
    plt.tight_layout()
    p4 = os.path.join(out_dir, "fig4_topology.png")
    plt.savefig(p4, dpi=130, bbox_inches='tight'); plt.close()
    print(f"[✓] {p4}")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    TOPOLOGIES = ["random", "grid", "geant"]
    N_FLOWS    = 25
    N_EPISODES = 4000   # augmenter pour de meilleurs résultats (5000–10000)

    all_results = []
    for topo in TOPOLOGIES:
        res = run_experiment(
            topo_name  = topo,
            n_flows    = N_FLOWS,
            n_episodes = N_EPISODES,
            lam_range  = (100, 600),
        )
        all_results.append(res)

    plot_all(all_results, out_dir=".")

    print("\n✅  Fichiers générés :")
    for f in ["fig1_bar_metrics.png", "fig2_improvement_pct.png",
              "fig3_convergence.png", "fig4_topology.png"]:
        print(f"   {f}")

    print("""
💡 Transfert vers un autre projet :
   from sdn_rl_v2 import QRoutingAgent, SDNNetwork, build_topology

   G   = build_topology('custom', edges=[(0,1),(1,2),(2,3),(0,3)])
   net = SDNNetwork(G, mu=600, K=20)
   ag  = QRoutingAgent(G, net)
   ag.load('qtable_random.pkl')       # réutiliser un modèle entraîné
   path = ag.get_path(src, dst, lam=200)
""")
