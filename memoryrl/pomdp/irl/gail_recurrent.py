# memoryrl/irl/gail_recurrent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RecurrentPolicyNet(nn.Module):
    """
    Recurrent stochastic policy π(a | s, h).
    Outputs logits over actions; you decide whether to sample or argmax.

    Supports GRU or LSTM core.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        core_type: Literal["gru", "lstm"] = "gru",
        hidden_dim: int = 128,
        num_layers: int = 1,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.core_type = core_type

        self.enc = nn.Linear(obs_dim, hidden_dim)

        if core_type == "gru":
            self.core = nn.GRU(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        elif core_type == "lstm":
            self.core = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("core_type must be 'gru' or 'lstm'")

        self.head = nn.Linear(hidden_dim, n_actions)

    def forward(
        self,
        obs: torch.Tensor,
        h=None,
    ):
        """
        obs : (batch, seq, obs_dim)
        h   : recurrent hidden state (GRU: tensor, LSTM: (h,c))

        Returns
        -------
        logits : (batch, seq, n_actions)
        h_new  : new hidden state
        """
        x = F.relu(self.enc(obs))
        out, h_new = self.core(x, h)
        logits = self.head(out)
        return logits, h_new


class Discriminator(nn.Module):
    """
    GAIL discriminator D_φ(s, a) ~ probability 'expert'.

    You can choose:
      - feed only s and a,
      - or (s, a, s') if you want; here we use (s, a).
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions

        self.obs_enc = nn.Linear(obs_dim, hidden_dim)
        self.act_enc = nn.Embedding(n_actions, hidden_dim)

        self.fc1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        obs     : (batch, obs_dim)
        actions : (batch,) int64 in [0, n_actions)

        Returns
        -------
        logits : (batch, 1)
        prob   : (batch, 1) sigmoid
        """
        x_obs = F.relu(self.obs_enc(obs))
        x_act = F.relu(self.act_enc(actions))

        x = torch.cat([x_obs, x_act], dim=-1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        prob = torch.sigmoid(logits)
        return logits, prob


@dataclass
class GAILConfig:
    obs_dim: int
    n_actions: int
    core_type: Literal["gru", "lstm"] = "gru"
    policy_hidden_dim: int = 128
    policy_num_layers: int = 1
    disc_hidden_dim: int = 128
    policy_lr: float = 3e-4
    disc_lr: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95  # if you want GAE


class GAILRecurrent:
    """
    Helper wrapper for GAIL-style training with a recurrent policy.

    This does NOT provide the full training loop (since you’ll plug it into
    your POMDP env & data pipeline), but encapsulates:

      - policy network πθ
      - discriminator Dφ
      - basic update steps for D and π

    You:
      - collect 'expert' and 'agent' trajectories,
      - build batches (s, a) for the discriminator,
      - and (s, a, returns/advantages) for the policy update.
    """

    def __init__(self, cfg: GAILConfig):
        self.cfg = cfg
        self.device = Device

        self.policy = RecurrentPolicyNet(
            obs_dim=cfg.obs_dim,
            n_actions=cfg.n_actions,
            core_type=cfg.core_type,
            hidden_dim=cfg.policy_hidden_dim,
            num_layers=cfg.policy_num_layers,
        ).to(self.device)

        self.discriminator = Discriminator(
            obs_dim=cfg.obs_dim,
            n_actions=cfg.n_actions,
            hidden_dim=cfg.disc_hidden_dim,
        ).to(self.device)

        self.optim_policy = torch.optim.Adam(self.policy.parameters(), lr=cfg.policy_lr)
        self.optim_disc = torch.optim.Adam(self.discriminator.parameters(), lr=cfg.disc_lr)

    # ---------- Policy interaction ----------

    def select_action(self, obs, h, greedy: bool = False):
        """
        Single-step action selection.

        obs : np.ndarray or torch.Tensor of shape (obs_dim,)
        h   : hidden state (GRU: tensor, LSTM: (h,c)) or None

        Returns
        -------
        action : int
        log_prob : float
        h_new : new hidden
        """
        if not torch.is_tensor(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)[None, None, :]
        else:
            obs_t = obs.to(self.device)[None, None, :]

        self.policy.eval()
        with torch.no_grad():
            logits, h_new = self.policy(obs_t, h)  # (1,1,n_actions)
            logits_last = logits[:, -1, :]         # (1,n_actions)
            probs = F.softmax(logits_last, dim=-1)

            if greedy:
                action = int(probs.argmax(dim=-1).item())
                log_prob = float(torch.log(probs.max(dim=-1).values + 1e-8).item())
            else:
                dist = torch.distributions.Categorical(probs=probs)
                a = dist.sample()
                action = int(a.item())
                log_prob = float(dist.log_prob(a).item())

        return action, log_prob, h_new

    # ---------- Discriminator update (GAIL core) ----------

    def update_discriminator(
        self,
        obs_expert: torch.Tensor,
        act_expert: torch.Tensor,
        obs_agent: torch.Tensor,
        act_agent: torch.Tensor,
    ) -> float:
        """
        One discriminator update step.

        obs_* : (B, obs_dim)
        act_* : (B,) int64
        """
        obs_expert = obs_expert.to(self.device)
        act_expert = act_expert.to(self.device).long()
        obs_agent = obs_agent.to(self.device)
        act_agent = act_agent.to(self.device).long()

        logits_expert, prob_expert = self.discriminator(obs_expert, act_expert)
        logits_agent, prob_agent = self.discriminator(obs_agent, act_agent)

        # standard GAIL loss:
        #   max_D E[log D(expert)] + E[log (1 - D(agent))]
        # so we minimize:
        loss_expert = F.binary_cross_entropy(prob_expert, torch.ones_like(prob_expert))
        loss_agent = F.binary_cross_entropy(prob_agent, torch.zeros_like(prob_agent))
        loss = loss_expert + loss_agent

        self.optim_disc.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
        self.optim_disc.step()

        return float(loss.item())

    # ---------- Policy update using discriminator's reward ----------

    def update_policy(
        self,
        obs_batch: torch.Tensor,
        act_batch: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_ratio: float = 0.2,
    ) -> float:
        """
        Simple PPO-style clipped policy update using discriminator-implied reward.

        obs_batch    : (B,T,obs_dim)
        act_batch    : (B,T)
        old_log_probs: (B,T)
        advantages   : (B,T)
        """
        obs_batch = obs_batch.to(self.device)
        act_batch = act_batch.to(self.device).long()
        old_log_probs = old_log_probs.to(self.device)
        advantages = advantages.to(self.device)

        self.policy.train()

        B, T, _ = obs_batch.shape
        logits, _ = self.policy(obs_batch)  # (B,T,n_actions)
        logits = logits.view(B * T, -1)
        acts = act_batch.view(B * T)
        adv = advantages.view(B * T)
        old_lp = old_log_probs.view(B * T)

        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - old_lp)

        # PPO clipped objective
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv
        loss_pi = -(torch.min(unclipped, clipped)).mean()

        self.optim_policy.zero_grad()
        loss_pi.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 10.0)
        self.optim_policy.step()

        return float(loss_pi.item())
