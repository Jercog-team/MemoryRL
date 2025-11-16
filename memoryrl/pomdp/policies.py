"""
Policies for the 8-port POMDP.

Core idea: a common BasePolicy interface, plus:

- GreedyMemoryPolicy: analytic policy reproducing your current
  hand-written greedy rule (f_theta * age - λ * penalty).

- DQNRecurrentPolicy / GAILRecurrentPolicy: thin wrappers around
  your recurrent agents, exposing the same interface.

All policies are defined in terms of *probabilities over 8 actions*
so they can be used both for simulation (sampling) and for
likelihood evaluation on real sequences (log_prob_of_action).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Dict, Any, Optional

import numpy as np


# ---------------------------------------------------------
# Base interface
# ---------------------------------------------------------

class BasePolicy(Protocol):
    """
    Abstract interface all policies must follow.

    The environment will pass a `state` dict with at least:
        - 't'            : trial index (int)
        - 'step'         : poke index within trial (int)
        - 'distance_bin' : 0..4
        - 'T_r'          : np.ndarray(8,), last visit trial per port (int, -1 if never)
        - 'f_theta'      : np.ndarray(8,), memory distribution over ports
        - 'r_cur'        : float, radial position
        - 'theta_cur'    : float, angular position (radians)
        - 'today_port'   : int, correct port index (0..7)
        - 'yesterday_port': int, yesterday's correct port (0..7)
        - 'tau_true'     : int or None (may be ignored by the policy)

    Policies may ignore keys they don’t need.
    """

    def reset_episode(self) -> None:
        """Reset any recurrent or episode-level state (e.g. RNN hidden state)."""
        ...

    def action_probs(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Return a vector of size 8 with probabilities over actions 0..7.
        Must be a proper probability distribution (sums to 1).
        """
        ...

    def log_prob_of_action(self, state: Dict[str, Any], action: int) -> float:
        """
        Return log π(a | state). Default implementation uses action_probs().
        Policies that implement ε-greedy with tie-breaking may want to
        override this for exact behaviour.
        """
        probs = np.asarray(self.action_probs(state), dtype=float)
        if probs.shape != (8,):
            raise ValueError("action_probs must return shape (8,).")
        a = int(action)
        if not (0 <= a < 8):
            return -np.inf
        p = max(float(probs[a]), 1e-12)
        return float(np.log(p))


# ---------------------------------------------------------
# Greedy memory-based policy (your current rule)
# ---------------------------------------------------------

@dataclass
class GreedyMemoryPolicy(BasePolicy):
    """
    Analytic greedy policy that reproduces your current POMDP behaviour:

        score_j ∝ f_theta[j] * age_j(t)  -  lambda_dist * penalty_j
        age_j(t) = max(t - T_r[j], 1)     (trial-based "time since visit")
        penalty_j = -2 * r_cur * cos(phi_j - theta_cur)

    Then we choose action with ε-greedy and random tie-breaking:
        - with prob ε: uniform over 8 ports
        - with prob (1-ε): uniform over argmax set of score_j

    eps_first is used at the first poke in a trial (step == 0),
    eps_rest otherwise. ky_vector and w_vector are indexed by
    distance_bin (0..4) to select the concentration / weight
    for yesterday’s trace.

    This class does NOT update the belief; it just uses the
    static f_theta and T_r passed via the state dict.
    """

    k_today: float
    ky_vector: np.ndarray           # shape (5,) for distance bins
    w_vector: np.ndarray            # shape (5,) for distance bins
    eps_first: float
    eps_rest: float
    lambda_dist: float
    port_angles: np.ndarray         # shape (8,), angles of ports (radians)

    def __post_init__(self) -> None:
        self.ky_vector = np.asarray(self.ky_vector, dtype=float)
        self.w_vector = np.asarray(self.w_vector, dtype=float)
        if self.ky_vector.shape[0] != 5 or self.w_vector.shape[0] != 5:
            raise ValueError("ky_vector and w_vector must have shape (5,) for 5 distance bins.")
        self.port_angles = np.asarray(self.port_angles, dtype=float)
        if self.port_angles.shape[0] != 8:
            raise ValueError("port_angles must have length 8.")
        self.eps_first = float(self.eps_first)
        self.eps_rest = float(self.eps_rest)
        self.lambda_dist = float(self.lambda_dist)

    # For the greedy analytic policy we have no internal recurrent state
    def reset_episode(self) -> None:
        pass

    def _epsilon_for_state(self, state: Dict[str, Any]) -> float:
        step = int(state.get("step", 0))
        return self.eps_first if step == 0 else self.eps_rest

    def _score_vector(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compute the pre-ε greedy scores (T_t_all_sim in your notebook),
        i.e. the vector used just before ε-greedy sampling.
        """
        t = int(state.get("t", 0))                 # trial index
        T_r = np.asarray(state["T_r"], dtype=float)  # last visit per port
        f_theta = np.asarray(state["f_theta"], dtype=float)  # shape (8,)
        r_cur = float(state["r_cur"])
        theta_cur = float(state["theta_cur"])
        # distance bin is useful to choose ky, w elsewhere, but here we
        # assume f_theta was already built from those; we only use f_theta.

        # age_j(t) = max(t - T_r[j], 1)
        age = (t - T_r).astype(float)
        age[age < 0] = 1.0

        # distance penalty on geometry
        penalty = -2.0 * r_cur * np.cos(self.port_angles - theta_cur)

        raw = f_theta * age
        total = float(raw.sum())
        if not np.isfinite(total) or total <= 0.0:
            score_term = np.zeros_like(raw)
        else:
            score_term = raw / total  # normalized "attraction" term

        scores = score_term - self.lambda_dist * penalty
        # Clean inf/nan
        scores = np.where(np.isfinite(scores), scores, -1e15)
        return scores

    def action_probs(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Return probabilities π(a | state) under ε-greedy with tie-breaking:

            - find argmax set of scores
            - with prob ε: uniform over 8
            - with prob 1-ε: uniform over argmax set
        """
        scores = self._score_vector(state)
        eps = self._epsilon_for_state(state)

        # Argmax set
        mx = np.max(scores)
        if not np.isfinite(mx):
            # degenerate, fallback to uniform
            return np.full(8, 1.0 / 8.0)

        ties = np.flatnonzero(np.isclose(scores, mx, atol=1e-12))
        if ties.size == 0:
            ties = np.array([int(np.argmax(scores))])
        m = float(ties.size)

        # Base uniform
        probs = np.full(8, eps / 8.0, dtype=float)
        # Add greedy mass to argmax set
        probs[ties] += (1.0 - eps) * (1.0 / m)

        # Numerical cleaning
        probs = np.clip(probs, 0.0, np.inf)
        s = probs.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(8, 1.0 / 8.0)
        return probs / s

    def log_prob_of_action(self, state: Dict[str, Any], action: int) -> float:
        """
        Exact log-prob under ε-greedy + random tie-breaking, matching your
        notebook derivation:

           p(a) = ε/8       if a not in argmax set
                = ε/8 + (1-ε)/|ties| if a in argmax set
        """
        scores = self._score_vector(state)
        eps = self._epsilon_for_state(state)
        a = int(action)
        if not (0 <= a < 8):
            return -np.inf

        mx = np.max(scores)
        if not np.isfinite(mx):
            # fallback to uniform
            return float(np.log(1.0 / 8.0))

        ties = np.flatnonzero(np.isclose(scores, mx, atol=1e-12))
        if ties.size == 0:
            ties = np.array([int(np.argmax(scores))])
        m = float(ties.size)

        in_ties = bool(a in ties)
        p = eps / 8.0 + ((1.0 - eps) / m if in_ties else 0.0)
        p = max(float(p), 1e-12)
        return float(np.log(p))


# ---------------------------------------------------------
# Wrappers for DQN / GAIL recurrent agents
# ---------------------------------------------------------

@dataclass
class DQNRecurrentPolicy(BasePolicy):
    """
    Thin wrapper around your recurrent DQN agent.

    The DQN model is expected to expose an API like:

        q_values, new_hidden = model.forward(obs, hidden)

    where obs is a 1D or 2D array with features built from `state`
    (belief, last visit, trial index, etc.), and q_values is a
    vector of length 8 (one per port, 0..7).

    This wrapper converts Q-values to a probability distribution
    via softmax (temperature optional), and uses that both to
    sample actions and to compute log-probabilities.
    """
    model: Any                       # your DQN model instance
    feature_fn: Any                  # callable: state -> np.ndarray(obs)
    temperature: float = 1.0
    hidden_state: Optional[Any] = None

    def reset_episode(self) -> None:
        self.hidden_state = None
        if hasattr(self.model, "reset_hidden"):
            self.model.reset_hidden()

    def _q_values_and_update(self, state: Dict[str, Any]) -> np.ndarray:
        obs = np.asarray(self.feature_fn(state), dtype=float)
        if obs.ndim == 1:
            obs_in = obs[None, :]  # (1, D)
        else:
            obs_in = obs
        if self.hidden_state is None:
            q, new_h = self.model.forward(obs_in)
        else:
            q, new_h = self.model.forward(obs_in, self.hidden_state)
        self.hidden_state = new_h
        q = np.asarray(q).reshape(-1)
        if q.shape[0] != 8:
            raise ValueError("DQN must output 8 Q-values (one per port).")
        return q

    def action_probs(self, state: Dict[str, Any]) -> np.ndarray:
        q = self._q_values_and_update(state)
        tau = float(self.temperature)
        if tau <= 0 or not np.isfinite(tau):
            # greedy limit
            probs = np.zeros_like(q)
            probs[np.argmax(q)] = 1.0
            return probs
        z = q / tau
        z = z - np.max(z)  # for stability
        ex = np.exp(z)
        s = ex.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(8, 1.0 / 8.0)
        return ex / s


@dataclass
class GAILRecurrentPolicy(BasePolicy):
    """
    Wrapper around a GAIL-style recurrent policy network.

    The GAIL model is expected to expose something like:

        pi, new_hidden = model.forward(obs, hidden)

    where `pi` is already a probability distribution over 8 ports.
    """
    model: Any
    feature_fn: Any                  # callable: state -> np.ndarray(obs)
    hidden_state: Optional[Any] = None

    def reset_episode(self) -> None:
        self.hidden_state = None
        if hasattr(self.model, "reset_hidden"):
            self.model.reset_hidden()

    def action_probs(self, state: Dict[str, Any]) -> np.ndarray:
        obs = np.asarray(self.feature_fn(state), dtype=float)
        if obs.ndim == 1:
            obs_in = obs[None, :]
        else:
            obs_in = obs
        if self.hidden_state is None:
            pi, new_h = self.model.forward(obs_in)
        else:
            pi, new_h = self.model.forward(obs_in, self.hidden_state)
        self.hidden_state = new_h
        pi = np.asarray(pi).reshape(-1)
        if pi.shape[0] != 8:
            raise ValueError("GAIL policy must output 8 probabilities (one per port).")
        pi = np.clip(pi, 0.0, np.inf)
        s = pi.sum()
        if s <= 0 or not np.isfinite(s):
            return np.full(8, 1.0 / 8.0)
        return pi / s


# Backwards-compat alias if you used `GreedyPolicy` before
GreedyPolicy = GreedyMemoryPolicy
