"""
Routing Engine — DDPG (Deep Deterministic Policy Gradient)
===========================================================
Implémentation exacte selon:
  Kim et al., "Deep Reinforcement Learning-Based Routing on SDN", IEEE Access 2022

Architecture:
  - État    : ATVM (Aggregated Traffic Volume Matrix) N×N normalisée par µmax
  - Action  : poids continus de chaque lien [w_min, w_max]
  - Reward  : R = α×rd + (1-α)×rp  (α=0.9)
  - Actor   : FC(400) → ReLU → FC(300) → ReLU → FC(|E|) → tanh
  - Critic  : FC(400) → ReLU → [concat action] → FC(300) → ReLU → FC(1)
  - Noise   : Ornstein-Uhlenbeck process
  - Routing : Dijkstra weighted shortest path
"""

import os
import logging
import numpy as np
import networkx as nx

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ml_models')
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Hyperparamètres (Table 2 & 3 du paper) ───────────────────────────────────
W_MIN        = 1        # poids minimum des liens
W_MAX        = 5        # poids maximum des liens
ALPHA        = 0.9      # pondération délai vs perte paquets
GAMMA        = 0.99     # facteur de discount
BATCH_SIZE   = 100      # taille du mini-batch (paper: 100)
BUFFER_SIZE  = 50_000   # taille du replay buffer (paper: 50,000)
LR_ACTOR     = 1e-5     # learning rate acteur (paper: 1e-5)
LR_CRITIC    = 1e-5     # learning rate critique (paper: 1e-5)
TAU_ACTOR    = 1e-5     # soft update acteur ε_a (paper: 1e-5)
TAU_CRITIC   = 1e-5     # soft update critique ε_c (paper: 1e-5)
HIDDEN1      = 400      # 1ère couche cachée (paper: 400)
HIDDEN2      = 300      # 2ème couche cachée (paper: 300)
WARMUP_STEPS = 100      # steps avant apprentissage (paper: 100)
MU_MAX       = 3000.0   # service rate max (paper: 3000 pkt/s)
K_CAPACITY   = 10_000   # capacité système switch (paper: 10,000)


# ── Ornstein-Uhlenbeck Noise ──────────────────────────────────────────────────

class OUNoise:
    """
    Ornstein-Uhlenbeck process pour exploration DDPG.
    Génère un bruit corrélé temporellement pour actions continues.
    """
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size  = size
        self.mu    = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + \
             self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state.copy()


# ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, s, a, r, ns):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (s, a, r, ns)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx   = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, ns = zip(*batch)
        return (np.array(s,  dtype=np.float32),
                np.array(a,  dtype=np.float32),
                np.array(r,  dtype=np.float32).reshape(-1, 1),
                np.array(ns, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


# ── Actor / Critic Networks ───────────────────────────────────────────────────

def _make_actor(state_dim, action_dim):
    try:
        import torch.nn as nn

        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(state_dim, HIDDEN1), nn.ReLU(),
                    nn.Linear(HIDDEN1, HIDDEN2),   nn.ReLU(),
                    nn.Linear(HIDDEN2, action_dim), nn.Tanh()
                )
                for m in self.net:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)

            def forward(self, s):
                # Scaler [-1,1] → [W_MIN, W_MAX]
                return W_MIN + (self.net(s) + 1.0) * 0.5 * (W_MAX - W_MIN)

        return Actor()
    except ImportError:
        return None


def _make_critic(state_dim, action_dim):
    try:
        import torch.nn as nn

        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1  = nn.Linear(state_dim, HIDDEN1)
                self.fc2  = nn.Linear(HIDDEN1 + action_dim, HIDDEN2)
                self.fc3  = nn.Linear(HIDDEN2, 1)
                self.relu = nn.ReLU()
                for m in [self.fc1, self.fc2, self.fc3]:
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

            def forward(self, s, a):
                import torch
                x = self.relu(self.fc1(s))
                x = self.relu(self.fc2(torch.cat([x, a], dim=1)))
                return self.fc3(x)

        return Critic()
    except ImportError:
        return None


# ── DDPG Agent ────────────────────────────────────────────────────────────────

class DDPGAgent:
    """
    Agent DDPG — Algorithm 1 du paper.
    Apprend à allouer les poids des liens pour minimiser
    délai end-to-end et perte de paquets.
    """

    def __init__(self, state_dim, action_dim):
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self._step      = 0
        self.buffer     = ReplayBuffer()
        self.noise      = OUNoise(action_dim)
        self._torch_ok  = False

        try:
            import torch
            import torch.optim as optim

            self._T      = torch
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.actor         = _make_actor(state_dim, action_dim).to(self._device)
            self.actor_target  = _make_actor(state_dim, action_dim).to(self._device)
            self.actor_target.load_state_dict(self.actor.state_dict())

            self.critic        = _make_critic(state_dim, action_dim).to(self._device)
            self.critic_target = _make_critic(state_dim, action_dim).to(self._device)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.actor_opt  = optim.Adam(self.actor.parameters(),  lr=LR_ACTOR)
            self.critic_opt = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

            self._torch_ok = True
            logger.info(f"[DDPG] Initialisé — device={self._device} "
                        f"state={state_dim} action={action_dim}")
        except (ImportError, Exception) as e:
            logger.warning(f"[DDPG] PyTorch indisponible: {e} — fallback Dijkstra")

    # Ligne 9 — Algorithm 1
    def select_action(self, state, add_noise=True):
        if not self._torch_ok:
            return np.ones(self.action_dim, dtype=np.float32)
        import torch
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self._device)
            a = self.actor(s).cpu().numpy()[0]
        if add_noise:
            a += self.noise.sample()
        return np.clip(a, W_MIN, W_MAX).astype(np.float32)

    # Ligne 11 — Algorithm 1
    def store(self, s, a, r, ns):
        self.buffer.push(s, a, r, ns)

    # Lignes 15-18 — Algorithm 1
    def learn(self):
        if not self._torch_ok or len(self.buffer) < WARMUP_STEPS:
            return None

        import torch, torch.nn.functional as F

        s, a, r, ns = self.buffer.sample(BATCH_SIZE)
        S  = torch.FloatTensor(s).to(self._device)
        A  = torch.FloatTensor(a).to(self._device)
        R  = torch.FloatTensor(r).to(self._device)
        NS = torch.FloatTensor(ns).to(self._device)

        # yi = R + γ × Q'(si+1, τ'(si+1))  — eq. 16
        with torch.no_grad():
            target_q = R + GAMMA * self.critic_target(NS, self.actor_target(NS))

        # Critic loss L(θQ) = (1/H) Σ (yi - Q(si,ai))²  — eq. 18
        critic_loss = F.mse_loss(self.critic(S, A), target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor loss ∇J(θτ) = -Q(s, τ(s))  — eq. 19
        actor_loss = -self.critic(S, self.actor(S)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Soft update θQ' = ε_c×θQ + (1-ε_c)×θQ'  — eq. 20
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(TAU_CRITIC * p.data + (1 - TAU_CRITIC) * pt.data)

        # Soft update θτ' = ε_a×θτ + (1-ε_a)×θτ'  — eq. 21
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(TAU_ACTOR * p.data + (1 - TAU_ACTOR) * pt.data)

        self._step += 1
        return float(critic_loss.item())

    def save(self, path=None):
        if not self._torch_ok:
            return
        path = path or os.path.join(MODELS_DIR, 'ddpg_routing.pt')
        self._T.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'step': self._step,
        }, path)
        logger.info(f"[DDPG] Sauvegardé → {path}")

    def load(self, path=None):
        if not self._torch_ok:
            return False
        path = path or os.path.join(MODELS_DIR, 'ddpg_routing.pt')
        if not os.path.exists(path):
            return False
        try:
            ck = self._T.load(path, map_location=self._device)
            self.actor.load_state_dict(ck['actor'])
            self.actor_target.load_state_dict(ck['actor_target'])
            self.critic.load_state_dict(ck['critic'])
            self.critic_target.load_state_dict(ck['critic_target'])
            self._step = ck.get('step', 0)
            logger.info(f"[DDPG] Chargé — step={self._step}")
            return True
        except Exception as e:
            logger.warning(f"[DDPG] Erreur chargement: {e}")
            return False


# ── Routing Engine ────────────────────────────────────────────────────────────

class RoutingEngine:
    """
    Moteur de routage DDPG + Dijkstra selon le paper.

    1. ATVM calculée depuis FlowStats (controller.py)
    2. DDPGAgent → poids optimaux des liens
    3. Dijkstra(poids) → chemin installé sur switches
    4. M/M/1/K → reward → DDPGAgent.learn()
    """

    def __init__(self, network_graph):
        self.network_graph = network_graph
        self.agent         = None
        self._initialized  = False
        self._last_state   = None
        self._last_action  = None
        self.flow_traffic  = {}   # {(src, dst): lambda pkt/s}

    def _init_agent(self):
        N = len(self.network_graph.graph.nodes())
        E = len(self.network_graph.graph.edges())
        if N < 2 or E == 0:
            return False
        self.agent = DDPGAgent(state_dim=N * N, action_dim=E)
        self.agent.load()
        self._initialized = True
        return True

    def get_path(self, src_dpid, dst_dpid):
        graph = self.network_graph.graph
        if src_dpid not in graph or dst_dpid not in graph:
            return None
        if src_dpid == dst_dpid:
            return [src_dpid]

        if not self._initialized:
            if not self._init_agent():
                return self._fallback(src_dpid, dst_dpid)

        # ATVM → action DDPG → poids liens
        state = self._compute_atvm().flatten()
        if self.agent and self.agent._torch_ok:
            weights = self.agent.select_action(state, add_noise=False)
            self._apply_weights(weights)
            self._last_state  = state
            self._last_action = weights

        return self._dijkstra(src_dpid, dst_dpid)

    def update_traffic(self, src_dpid, dst_dpid, lambda_rate):
        """Mise à jour du trafic agrégé — appelé par controller.py."""
        self.flow_traffic[(src_dpid, dst_dpid)] = lambda_rate

    def feedback(self, reward):
        """Reward M/M/1/K → apprentissage DDPG."""
        if self._last_state is None or not self.agent:
            return
        next_state = self._compute_atvm().flatten()
        self.agent.store(self._last_state, self._last_action, reward, next_state)
        loss = self.agent.learn()
        if loss and self.agent._step % 100 == 0:
            logger.info(f"[DDPG] step={self.agent._step} loss={loss:.5f}")
        self._last_state = self._last_action = None

    def recompute_all_paths(self):
        self._initialized = False
        if self.agent:
            self.agent.save()

    def get_all_paths(self, src, dst, k=3):
        try:
            return list(nx.shortest_simple_paths(
                self.network_graph.graph, src, dst, weight='weight'))[:k]
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _compute_atvm(self):
        """
        ATVM eq. 9-10: s_t^{i,j} = min(1, (1/µmax) × Σ λ^k × x^k_{ij})
        """
        nodes = sorted(self.network_graph.graph.nodes())
        N     = len(nodes)
        idx   = {n: i for i, n in enumerate(nodes)}
        atvm  = np.zeros((N, N), dtype=np.float32)
        for (src, dst), lam in self.flow_traffic.items():
            if src in idx and dst in idx:
                atvm[idx[src]][idx[dst]] += lam / MU_MAX
        return np.clip(atvm, 0.0, 1.0)

    def _apply_weights(self, weights):
        for k, (u, v) in enumerate(self.network_graph.graph.edges()):
            if k < len(weights):
                self.network_graph.graph[u][v]['weight'] = float(
                    np.clip(weights[k], W_MIN, W_MAX))

    def _dijkstra(self, src, dst):
        try:
            return nx.dijkstra_path(
                self.network_graph.graph, src, dst, weight='weight')
        except Exception:
            return self._fallback(src, dst)

    def _fallback(self, src, dst):
        try:
            return nx.shortest_path(self.network_graph.graph, src, dst)
        except Exception:
            return None
