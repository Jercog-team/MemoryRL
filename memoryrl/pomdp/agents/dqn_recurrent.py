# memoryrl/agents/dqn_recurrent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class QRecurrentNetGRU(nn.Module):
    """
    Q-network with a GRU core.
    Input: (batch, seq, obs_dim)
    Output: Q-values (batch, seq, n_actions) and hidden state.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.enc = nn.Linear(obs_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        obs: torch.Tensor,
        h: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        obs : (batch, seq, obs_dim)
        h   : (num_layers, batch, hidden_dim) or None

        Returns
        -------
        q : (batch, seq, n_actions)
        h_new : (num_layers, batch, hidden_dim)
        """
        x = F.relu(self.enc(obs))
        q, h_new = self.gru(x, h)
        q = self.head(q)
        return q, h_new


class QRecurrentNetLSTM(nn.Module):
    """
    Q-network with an LSTM core.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.enc = nn.Linear(obs_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        obs: torch.Tensor,
        h: Tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        obs : (batch, seq, obs_dim)
        h   : (h0, c0) each of shape (num_layers, batch, hidden_dim) or None

        Returns
        -------
        q    : (batch, seq, n_actions)
        h_new: (h, c) same shapes as input h
        """
        x = F.relu(self.enc(obs))
        q, h_new = self.lstm(x, h)
        q = self.head(q)
        return q, h_new


@dataclass
class DQNRecurrentConfig:
    obs_dim: int
    n_actions: int
    core_type: Literal["gru", "lstm"] = "gru"
    hidden_dim: int = 128
    num_layers: int = 1
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.005  # soft update for target net
    eps_start: float = 1.0
    eps_final: float = 0.1
    eps_decay: int = 50_000  # steps


class DQNRecurrentAgent:
    """
    Recurrent DQN agent (GRU or LSTM).
    - You decide what the observation vector is (belief, distances, etc.).
    - Handles epsilon-greedy selection given Q-values.
    - Training: you feed sequences (batch, seq_len, obs_dim) yourself.

    This is intentionally 'env-agnostic': you can use it with your POMDP
    simulation or a gym-style env.
    """

    def __init__(self, cfg: DQNRecurrentConfig):
        self.cfg = cfg
        self.device = Device

        if cfg.core_type == "gru":
            self.q_net = QRecurrentNetGRU(
                obs_dim=cfg.obs_dim,
                n_actions=cfg.n_actions,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
            ).to(self.device)
            self.q_target = QRecurrentNetGRU(
                obs_dim=cfg.obs_dim,
                n_actions=cfg.n_actions,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
            ).to(self.device)
        elif cfg.core_type == "lstm":
            self.q_net = QRecurrentNetLSTM(
                obs_dim=cfg.obs_dim,
                n_actions=cfg.n_actions,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
            ).to(self.device)
            self.q_target = QRecurrentNetLSTM(
                obs_dim=cfg.obs_dim,
                n_actions=cfg.n_actions,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
            ).to(self.device)
        else:
            raise ValueError("core_type must be 'gru' or 'lstm'")

        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=cfg.lr)

        self.total_steps = 0

    def epsilon(self):
        """Linear epsilon decay."""
        frac = min(self.total_steps / self.cfg.eps_decay, 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_final - self.cfg.eps_start)

    def select_action(
        self,
        obs: np.ndarray,
        h,
        greedy: bool = False,
    ):
        """
        Single-step action selection.

        Parameters
        ----------
        obs : np.ndarray, shape (obs_dim,)
        h   : hidden state (GRU: tensor, LSTM: (h,c)) or None
        greedy : bool
            If True, ignore epsilon and pick argmax.

        Returns
        -------
        action : int in [0, n_actions)
        h_new  : updated hidden state
        """
        self.total_steps += 1
        eps = self.epsilon()
        if greedy:
            eps = 0.0

        if np.random.rand() < eps:
            # exploration
            action = np.random.randint(self.cfg.n_actions)
            # propagate hidden with dummy step if you want; often we just reuse h
            return action, h

        self.q_net.eval()
        with torch.no_grad():
            o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, None, :]
            q, h_new = self.q_net(o, h)
            q_last = q[:, -1]  # (1, n_actions)
            action = int(q_last.argmax(dim=-1).item())

        return action, h_new

    def soft_update_target(self):
        """Soft-update of target network parameters."""
        with torch.no_grad():
            for p, p_t in zip(self.q_net.parameters(), self.q_target.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def train_step(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_rewards: torch.Tensor,
        batch_next_obs: torch.Tensor,
        batch_dones: torch.Tensor,
        h0,
        h0_next,
    ) -> float:
        """
        One DQN update on a batch of sequences.

        Shapes
        ------
        batch_obs      : (B, T, obs_dim)
        batch_actions  : (B, T)      int64
        batch_rewards  : (B, T)      float32
        batch_next_obs : (B, T, obs_dim)
        batch_dones    : (B, T)      float32, 1 if done else 0
        h0, h0_next    : hidden state tensors for q_net and q_target respectively.

        Returns
        -------
        loss_value : float
        """
        self.q_net.train()

        batch_obs = batch_obs.to(self.device)
        batch_next_obs = batch_next_obs.to(self.device)
        batch_actions = batch_actions.to(self.device).long()
        batch_rewards = batch_rewards.to(self.device)
        batch_dones = batch_dones.to(self.device)

        # forward current Q
        q, _ = self.q_net(batch_obs, h0)  # (B, T, n_actions)
        q = q.gather(-1, batch_actions.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # target Q
        with torch.no_grad():
            q_next, _ = self.q_target(batch_next_obs, h0_next)
            q_next_max = q_next.max(dim=-1).values  # (B, T)
            target = batch_rewards + self.cfg.gamma * (1.0 - batch_dones) * q_next_max

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        self.soft_update_target()

        return float(loss.item())
