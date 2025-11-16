"""
POMDP simulation code for the 8-port maze.

Key entry point:
    simulate_sessions_with_policy(...)

It is policy-agnostic: you pass any `BasePolicy` instance
(GreedyMemoryPolicy, DQNRecurrentPolicy, GAILRecurrentPolicy),
and the environment handles the geometry, reward, and history.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Tuple, Optional, Callable

import numpy as np

from .policies import BasePolicy
from .belief import initialize_belief   # <- your existing function


@dataclass
class EnvData:
    """
    Bundle all environment-side arrays that are indexed per-session.

    All sequences must be in the SAME canonical session order.
    """
    positionsStartTrial_polar: Sequence[np.ndarray]  # list[session] of (T,2)
    distance_seq: np.ndarray                         # (n_sessions,), ints 0..4
    correct_ports: np.ndarray                        # (n_sessions,), 1..8
    yesterday_ports: np.ndarray                      # (n_sessions,), 1..8
    tau_true_per_session: np.ndarray                 # (n_sessions,), int or -1 if unknown


def build_session_order(distance_seq: Sequence[int], oversample_times: int = 1) -> np.ndarray:
    """
    Reproduce your oversampling trick: sessions with distance_bin == 0
    are repeated `oversample_times` times, others appear once.
    """
    distance_seq = np.asarray(distance_seq, dtype=int)
    order = []
    for s in range(len(distance_seq)):
        if int(distance_seq[s]) == 0:
            order.extend([s] * oversample_times)
        else:
            order.append(s)
    return np.array(order, dtype=int)


def simulate_sessions_with_policy(
    params_mem: Dict[str, Any],
    policy: BasePolicy,
    env: EnvData,
    port_angles: np.ndarray,
    max_pokes_per_trial: int = 4,
    session_order: Optional[np.ndarray] = None,
    rng_seed: int = 123,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate sessions in the POMDP using the given policy.

    Parameters
    ----------
    params_mem : dict
        Memory parameters passed to `initialize_belief`, e.g.:
            {
                "k_today": float,
                "ky_vector": np.array(5),
                "w_vector": np.array(5),
            }
        Only the keys needed by your initialize_belief are required.
    policy : BasePolicy
        Any policy implementing BasePolicy (greedy, DQN, GAIL...).
    env : EnvData
        Environment data (positions, correct/yesterday ports, tau_true).
    port_angles : np.ndarray
        Array shape (8,) with angles of each port in radians.
    max_pokes_per_trial : int
        Maximum number of pokes per trial.
    session_order : np.ndarray or None
        Optional array of session indices specifying the order
        for simulation (for oversampling). If None, sessions
        are used in canonical order 0..N-1.
    rng_seed : int
        Seed for NumPy random generator.

    Returns
    -------
    sim_pokes_sessions : np.ndarray (object)
        sessions -> list-of-trials -> list-of-pokes (0..7).
    hist_per_trial : np.ndarray (object)
        sessions -> np.ndarray(num_trials, 8), per-trial histograms.
    hist_per_session : np.ndarray
        (num_sessions_sim, 8), per-session histograms.
    """
    rng = np.random.default_rng(rng_seed)
    port_angles = np.asarray(port_angles, dtype=float)
    n_sessions = len(env.positionsStartTrial_polar)

    if session_order is None:
        session_order = np.arange(n_sessions, dtype=int)
    else:
        session_order = np.asarray(session_order, dtype=int)

    # unpack memory params, compatible with your initialize_belief
    k_today = float(params_mem["k_today"])
    ky_vector = np.asarray(params_mem["ky_vector"], dtype=float)  # shape (5,)
    w_vector = np.asarray(params_mem["w_vector"], dtype=float)    # shape (5,)

    sim_pokes_sessions = []
    hist_per_trial = []
    hist_per_session = []

    for sim_idx, sess_idx in enumerate(session_order):
        distance_bin = int(env.distance_seq[sess_idx])   # 0..4
        r_today = int(env.correct_ports[sess_idx])       # 1..8
        r_yest = int(env.yesterday_ports[sess_idx])      # 1..8
        tau_true = int(env.tau_true_per_session[sess_idx])

        # Build memory mixture for THIS session via your initialize_belief
        belief_init, f_theta, _ = initialize_belief(
            episode=sess_idx,
            ports_true=env.correct_ports,
            lag_ports_true=env.yesterday_ports,
            kappa_today=k_today,
            kappa_yesterday=float(ky_vector[distance_bin]),
            w_today=float(w_vector[distance_bin]),
        )
        # f_theta is 1D (8,). We use it directly in GreedyMemoryPolicy.

        # Last-visit time per port (trial index, -1 = never)
        T_r = np.full(8, -1, dtype=int)

        # Per-session storage
        session_trials = []          # list of list-of-pokes
        session_hist = np.zeros(8, dtype=int)
        trial_hist_list = []         # list of (8,) arrays

        # positionsStartTrial_polar[sess] has shape (T,2) (r,theta)
        start_polar = np.asarray(env.positionsStartTrial_polar[sess_idx], dtype=float)
        T_max_ep = start_polar.shape[0]
        reward = 0

        policy.reset_episode()

        # ---- Trial loop ----
        for t in range(T_max_ep):
            r_cur, theta_cur = float(start_polar[t, 0]), float(start_polar[t, 1])

            trial_pokes = []
            trial_hist = np.zeros(8, dtype=int)

            # ---- Poke loop ----
            num_pokes = rng.integers(0, max_pokes_per_trial + 1)
            for step in range(num_pokes):
                # Build state dict for the policy
                state = {
                    "t": t,
                    "step": step,
                    "distance_bin": distance_bin,
                    "T_r": T_r,
                    "f_theta": f_theta,
                    "r_cur": r_cur,
                    "theta_cur": theta_cur,
                    "today_port": r_today - 1,       # to 0..7
                    "yesterday_port": r_yest - 1,    # to 0..7
                    "tau_true": tau_true,
                }

                probs = np.asarray(policy.action_probs(state), dtype=float)
                # sample action 0..7
                if not np.all(np.isfinite(probs)) or probs.sum() <= 0:
                    probs = np.full(8, 1.0 / 8.0)
                else:
                    probs = probs / probs.sum()
                p = int(rng.choice(8, p=probs))

                # update last visit
                T_r[p] = t

                trial_pokes.append(p)
                trial_hist[p] += 1
                session_hist[p] += 1

                # Reward condition: same as your code
                if (p == r_today - 1) and (t >= tau_true):
                    reward = 1
                    break

                # Mouse is now at that port
                r_cur = 1.0
                theta_cur = port_angles[p]

            session_trials.append(trial_pokes)
            trial_hist_list.append(trial_hist)

            if reward == 1:
                break

        if not trial_hist_list:
            trial_hist_list.append(np.zeros(8, dtype=int))

        sim_pokes_sessions.append(np.array(session_trials, dtype=object))
        hist_per_trial.append(np.vstack(trial_hist_list))
        hist_per_session.append(session_hist)

    sim_pokes_sessions = np.array(sim_pokes_sessions, dtype=object)
    hist_per_trial = np.array(hist_per_trial, dtype=object)
    hist_per_session = np.vstack(hist_per_session) if hist_per_session else np.zeros((0, 8))

    return sim_pokes_sessions, hist_per_trial, hist_per_session
