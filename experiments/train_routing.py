"""
train_routing.py — Entraînement DDPG offline via modèle M/M/1/K
================================================================
Implémentation exacte selon:
  Kim et al., "Deep Reinforcement Learning-Based Routing on SDN", IEEE Access 2022

Le paper utilise un modèle réseau M/M/1/K pour entraîner le DDPG
en offline, sans dégrader le réseau réel pendant l'apprentissage.

Modèle M/M/1/K:
  - Arrivées Poisson avec taux λn(t)
  - Service exponentiel avec taux µn
  - Capacité système Kn (buffer limité)
  - Calcule: délai E[dn], perte Pb, utilisation ρn

Reward (eq. 14):
  R = α × rd + (1-α) × rp
  rd = 1 - D_avg / D_max      (délai normalisé)
  rp = 1 - L_tot / Σλn        (perte normalisée)

Usage:
  python experiments/train_routing.py
  python experiments/train_routing.py --iterations 200000 --switches 6
"""

import os
import sys
import time
import logging
import argparse
import random
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from controllers.routing import DDPGAgent, OUNoise, W_MIN, W_MAX, ALPHA, MU_MAX, K_CAPACITY

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(ROOT, 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)


# ── Modèle réseau M/M/1/K ────────────────────────────────────────────────────

class MM1KModel:
    """
    Modèle réseau M/M/1/K selon Section III-B du paper.

    Pour chaque switch n:
      ρn = λn / µn                        (intensité trafic)
      Pb_n = (1-ρn)×ρn^K / (1-ρn^(K+1))  (prob. perte, eq. 2)
      E[Nn] = ρn/(1-ρn) - (K+1)ρn^(K+1)/(1-ρn^(K+1))  (eq. 3)
      E[dn] = E[Nn] / (λn × (1 - Pb_n))   (délai, eq. 1)
    """

    def __init__(self, n_switches, mu=3000.0, K=10000):
        self.N   = n_switches
        self.mu  = mu    # service rate (pkt/s) — paper: 3000
        self.K   = K     # capacité système — paper: 10,000

    def compute_switch_metrics(self, lambda_n: float):
        """
        Calcule les métriques d'un switch avec taux d'arrivée λn.
        Retourne (delay, loss_prob, utilization).
        """
        mu = self.mu
        K  = self.K

        rho = lambda_n / mu  # ρ = λ/µ

        if rho <= 0:
            return 0.0, 0.0, 0.0

        # Probabilité de perte Pb (eq. 2)
        if abs(rho - 1.0) < 1e-9:   # ρ = 1 (cas limite)
            Pb    = 1.0 / (K + 1.0)
            E_N   = K / 2.0
        else:
            num_pb = (1 - rho) * (rho ** K)
            den_pb = 1 - rho ** (K + 1)
            Pb     = num_pb / (den_pb + 1e-12)

            # Occupation de file E[N] (eq. 3)
            E_N = (rho / (1 - rho)) - \
                  ((K + 1) * rho ** (K + 1)) / (1 - rho ** (K + 1) + 1e-12)

        E_N = max(0.0, E_N)
        Pb  = np.clip(Pb, 0.0, 1.0)

        # Délai moyen E[d] (eq. 1)
        effective_lambda = lambda_n * (1 - Pb)
        if effective_lambda < 1e-6:
            delay = 0.0
        else:
            delay = E_N / effective_lambda

        return delay, Pb, rho

    def compute_network_metrics(self, lambda_per_switch: dict, paths: list):
        """
        Calcule les métriques réseau globales.
        lambda_per_switch: {switch_id: lambda_n}
        paths: liste des chemins actifs

        Retourne (D_avg_e2e, L_tot, lambda_tot)
        """
        # Délai end-to-end par flow (eq. 4)
        total_delay  = 0.0
        n_flows      = len(paths)

        for path in paths:
            flow_delay = sum(
                self.compute_switch_metrics(
                    lambda_per_switch.get(sw, 10.0))[0]
                for sw in path
            )
            total_delay += flow_delay

        # Délai moyen (eq. 5)
        D_avg = total_delay / max(n_flows, 1)

        # Perte totale (eq. 7)
        L_tot     = 0.0
        lambda_tot = 0.0
        for sw, lam in lambda_per_switch.items():
            _, Pb, _ = self.compute_switch_metrics(lam)
            L_tot     += lam * Pb
            lambda_tot += lam

        return D_avg, L_tot, lambda_tot


# ── Environnement SDN avec M/M/1/K ───────────────────────────────────────────

class SDNEnvironment:
    """
    Environnement SDN simulé pour entraînement DDPG offline.
    Utilise M/M/1/K pour calculer état et reward sans toucher
    au réseau réel (Section III-A du paper).

    Topologies supportées: linéaire, grille, aléatoire.
    """

    def __init__(self, n_switches=6, topology='linear'):
        self.N        = n_switches
        self.mm1k     = MM1KModel(n_switches)
        self._build_topology(topology)
        self.n_links  = len(list(self.graph.edges()))
        self.state_dim  = n_switches * n_switches  # ATVM N×N
        self.action_dim = self.n_links

        # Demandes de trafic (λk entre 10 et 300 pkt/s — paper)
        self.n_flows    = max(10, n_switches * 2)
        self.flows      = []   # [(src, dst, lambda_k)]

    def _build_topology(self, topology):
        import networkx as nx
        self.graph = nx.DiGraph()

        for i in range(1, self.N + 1):
            self.graph.add_node(i)

        if topology == 'linear':
            for i in range(1, self.N):
                self.graph.add_edge(i, i + 1, weight=1.0)
                self.graph.add_edge(i + 1, i, weight=1.0)
            # Liens alternatifs
            if self.N >= 4:
                self.graph.add_edge(1, 3, weight=1.0)
                self.graph.add_edge(3, 1, weight=1.0)
            if self.N >= 6:
                self.graph.add_edge(2, 5, weight=1.0)
                self.graph.add_edge(5, 2, weight=1.0)
                self.graph.add_edge(1, 4, weight=1.0)
                self.graph.add_edge(4, 1, weight=1.0)

        elif topology == 'grid':
            # Grille (paper: 5×5 = 25 switches)
            side = int(np.ceil(np.sqrt(self.N)))
            for i in range(self.N):
                r, c = divmod(i, side)
                n = i + 1
                if c + 1 < side and i + 1 < self.N:
                    self.graph.add_edge(n, n + 1, weight=1.0)
                    self.graph.add_edge(n + 1, n, weight=1.0)
                if r + 1 < side and i + side < self.N:
                    self.graph.add_edge(n, n + side, weight=1.0)
                    self.graph.add_edge(n + side, n, weight=1.0)

    def reset(self):
        """Réinitialise les demandes de trafic (synthetic traffic)."""
        nodes = list(self.graph.nodes())
        self.flows = []
        for _ in range(self.n_flows):
            src = random.choice(nodes)
            dst = random.choice([n for n in nodes if n != src])
            lam = random.uniform(10, 300)   # paper: [10, 300] pkt/s
            self.flows.append((src, dst, lam))
        return self._compute_state(self._get_link_weights())

    def step(self, action: np.ndarray):
        """
        Applique les poids DDPG et calcule reward M/M/1/K.
        action: vecteur de poids pour chaque lien
        """
        # Appliquer les poids au graphe
        weights = np.clip(action, W_MIN, W_MAX)
        for k, (u, v) in enumerate(self.graph.edges()):
            if k < len(weights):
                self.graph[u][v]['weight'] = float(weights[k])

        # Calculer les chemins Dijkstra avec ces poids
        paths          = self._compute_paths()
        lambda_per_sw  = self._compute_lambda_per_switch(paths)

        # Métriques M/M/1/K
        D_avg, L_tot, lambda_tot = self.mm1k.compute_network_metrics(
            lambda_per_sw, paths)

        # Reward R = α×rd + (1-α)×rp  (eq. 14)
        reward = self._compute_reward(D_avg, L_tot, lambda_tot)

        # Changer légèrement le trafic pour dynamisme
        self._perturb_traffic()

        next_state = self._compute_state(weights)
        return next_state, reward

    def _compute_state(self, weights) -> np.ndarray:
        """
        ATVM normalisée (eq. 9-10).
        s_t^{i,j} = min(1, (1/µmax) × Σ λ^k × x^k_{ij})
        """
        nodes = sorted(self.graph.nodes())
        idx   = {n: i for i, n in enumerate(nodes)}
        N     = len(nodes)
        atvm  = np.zeros((N, N), dtype=np.float32)

        for src, dst, lam in self.flows:
            try:
                import networkx as nx
                path = nx.dijkstra_path(self.graph, src, dst, weight='weight')
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    if u in idx and v in idx:
                        atvm[idx[u]][idx[v]] += lam / MU_MAX
            except Exception:
                pass

        return np.clip(atvm.flatten(), 0.0, 1.0)

    def _compute_paths(self):
        """Calcule les chemins Dijkstra pour tous les flows."""
        import networkx as nx
        paths = []
        for src, dst, lam in self.flows:
            try:
                p = nx.dijkstra_path(self.graph, src, dst, weight='weight')
                paths.append(p)
            except Exception:
                paths.append([src, dst])
        return paths

    def _compute_lambda_per_switch(self, paths):
        """Calcule le taux d'arrivée agrégé par switch (λn)."""
        lambda_sw = {}
        for i, (src, dst, lam) in enumerate(self.flows):
            if i < len(paths):
                for sw in paths[i]:
                    lambda_sw[sw] = lambda_sw.get(sw, 0.0) + lam
        return lambda_sw

    def _compute_reward(self, D_avg, L_tot, lambda_tot):
        """
        R(st, at) = α × rd(t) + (1-α) × rp(t)  — eq. 14

        rd = 1 - D_avg / D_max  (délai normalisé, eq. 12)
        rp = 1 - L_tot / Σλn    (perte normalisée, eq. 13)
        """
        # D_max = délai si tous les paquets traversent le chemin le plus long
        # = Σ (Kn/µn) sur le chemin le plus long
        D_max = self.N * (K_CAPACITY / MU_MAX)
        rd    = 1.0 - np.clip(D_avg / (D_max + 1e-9), 0.0, 1.0)

        rp = 1.0 - np.clip(L_tot / (lambda_tot + 1e-9), 0.0, 1.0)

        R = ALPHA * rd + (1 - ALPHA) * rp
        return float(np.clip(R, 0.0, 1.0))

    def _perturb_traffic(self):
        """Perturbe légèrement les demandes de trafic (dynamisme)."""
        for i in range(len(self.flows)):
            src, dst, lam = self.flows[i]
            lam = np.clip(lam + random.uniform(-20, 20), 10, 300)
            self.flows[i] = (src, dst, lam)

    def _get_link_weights(self):
        return np.array([d['weight'] for _, _, d in self.graph.edges(data=True)])


# ── Boucle d'entraînement DDPG ───────────────────────────────────────────────

def train(n_iterations=200_000,
          n_switches=6,
          topology='linear',
          save_every=10_000):
    """
    Boucle d'entraînement offline DDPG selon Algorithm 1.
    """
    env   = SDNEnvironment(n_switches=n_switches, topology=topology)
    agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim)

    if not agent._torch_ok:
        logger.error("PyTorch requis. pip install torch")
        return None

    logger.info(f"\n{'='*55}")
    logger.info(f"Entraînement DDPG offline (Kim et al., IEEE Access 2022)")
    logger.info(f"Switches: {n_switches} | Liens: {env.n_links} | "
                f"State: {env.state_dim} | Action: {env.action_dim}")
    logger.info(f"Itérations: {n_iterations:,} | Topologie: {topology}")
    logger.info(f"α={ALPHA} | γ=0.99 | batch={100} | lr=1e-5")
    logger.info(f"{'='*55}\n")

    state          = env.reset()
    rewards_hist   = []
    losses_hist    = []
    best_reward    = -float('inf')
    window         = 500   # moving average (paper: 500 steps)

    t_start = time.time()

    for t in range(1, n_iterations + 1):

        # Ligne 9 — sélection action avec bruit OU
        action = agent.select_action(state, add_noise=True)

        # Ligne 10 — appliquer sur modèle réseau M/M/1/K
        next_state, reward = env.step(action)

        # Ligne 11 — stocker dans B
        agent.store(state, action, reward, next_state)

        # Lignes 15-18 — apprentissage
        loss = agent.learn()

        state = next_state
        rewards_hist.append(reward)

        if loss:
            losses_hist.append(loss)

        # Réinitialiser trafic périodiquement (nouveaux flows synthétiques)
        if t % 1000 == 0:
            state = env.reset()
            agent.noise.reset()

        # Sauvegarder meilleur modèle
        avg_r = np.mean(rewards_hist[-window:]) if len(rewards_hist) >= window else np.mean(rewards_hist)
        if avg_r > best_reward and len(rewards_hist) >= window:
            best_reward = avg_r
            agent.save(os.path.join(MODELS_DIR, 'ddpg_routing_best.pt'))

        # Log tous les 10,000 steps (paper: moving avg 500 steps)
        if t % 10_000 == 0:
            avg_loss = np.mean(losses_hist[-500:]) if losses_hist else 0.0
            elapsed  = time.time() - t_start
            logger.info(
                f"[{t:>7,}/{n_iterations:,}] "
                f"Reward(avg{window})={avg_r:.4f} | "
                f"Best={best_reward:.4f} | "
                f"Loss={avg_loss:.5f} | "
                f"t={elapsed:.0f}s"
            )

        if t % save_every == 0:
            agent.save()

    # Sauvegarde finale
    agent.save()

    final_avg = np.mean(rewards_hist[-window:])
    print(f"""
╔══════════════════════════════════════════════════════════╗
║       ENTRAÎNEMENT DDPG ROUTING TERMINÉ ✓               ║
╠══════════════════════════════════════════════════════════╣
║  Itérations    : {n_iterations:<37,}║
║  Reward final  : {final_avg:<37.4f}║
║  Meilleur      : {best_reward:<37.4f}║
║  Switches      : {n_switches:<37d}║
║  Liens         : {env.n_links:<37d}║
║                                                          ║
║  Modèles:                                               ║
║    ml_models/ddpg_routing.pt      (final)               ║
║    ml_models/ddpg_routing_best.pt (meilleur)            ║
╚══════════════════════════════════════════════════════════╝
""")
    return agent


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='DDPG Routing Trainer — Kim et al. 2022')
    p.add_argument('--iterations', type=int, default=200_000,
                   help='Nombre d\'itérations (paper: 200,000)')
    p.add_argument('--switches',   type=int, default=6,
                   help='Nombre de switches (paper: 6 pour linear)')
    p.add_argument('--topology',   type=str, default='linear',
                   choices=['linear', 'grid'],
                   help='Topologie (linear ou grid)')
    p.add_argument('--save-every', type=int, default=10_000,
                   help='Sauvegarder tous les N steps')
    args = p.parse_args()

    train(
        n_iterations=args.iterations,
        n_switches=args.switches,
        topology=args.topology,
        save_every=args.save_every,
    )
