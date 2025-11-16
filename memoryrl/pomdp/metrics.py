# pomdp/metrics.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch

from .belief import update_belief, initialize_belief
from .policy import compute_port_scores
from .simulation import MemoryAgent
from .agents.dqn_recurrent import DQNRecurrentAgent
from .irl.gail_recurrent import GAILRecurrent


# ---------------------------------------------------------------------- #
# Helper: heuristic policy probabilities                                 #
# ---------------------------------------------------------------------- #

def heuristic_action_probs(
    prior_ports: np.ndarray,
    last_visit_trials: np.ndarray,
    trial_index: int,
    step_in_trial: int,
    r_cur: float,
    theta_cur: float,
    port_angles: np.ndarray,
    lambda_dist: float,
    eps_first: float,
    eps_rest: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Compute π(a | s_t) for your heuristic epsilon-greedy policy.

    This matches the logic of epsilon_greedy_policy, but instead of sampling
    an action, we return the full probability vector over actions.

    π(a) = ε / N + (1-ε) * I[a in argmax(scores)] / |argmax(scores)|
    """
    port_angles = np.asarray(port_angles, dtype=float)
    prior_ports = np.asarray(prior_ports, dtype=float)
    last_visit_trials = np.asarray(last_visit_trials, dtype=int)

    # belief f_t(r) via update_belief
    f_t = update_belief(
        prior_ports=prior_ports,
        last_visit_trials=last_visit_trials,
        trial_index=trial_index,
        reward=None,
    )

    # scores via the same formula used in your policy
    scores = compute_port_scores(
        prior_ports=prior_ports,
        last_visit_trials=last_visit_trials,
        trial_index=trial_index,
        r_cur=r_cur,
        theta_cur=theta_cur,
        port_angles=port_angles,
        lambda_dist=lambda_dist,
    )

    n_ports = scores.shape[0]
    eps = eps_first if step_in_trial == 0 else eps_rest

    mx = scores.max()
    ties = np.flatnonzero(np.isclose(scores, mx, atol=1e-12))

    probs = np.ones(n_ports, dtype=float) * (eps / n_ports)
    if len(ties) > 0:
        probs[ties] += (1.0 - eps) / len(ties)

    return probs


# ---------------------------------------------------------------------- #
# 1) LL for heuristic policy                                             #
# ---------------------------------------------------------------------- #

def sequence_loglik_heuristic(
    sessions_pokes: List[List[np.ndarray]],
    ports_true_all: np.ndarray,
    lag_ports_true_all: np.ndarray,
    distance_seq_all: np.ndarray,
    positionsStartTrial_polar_all: List[np.ndarray],
    k_today: float,
    ky_vector: List[float],
    w_vector: List[float],
    eps_first: float,
    eps_rest: float,
    lambda_dist: float,
    port_angles: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Compute log-likelihood of REAL sequences under your heuristic policy.

    Parameters
    ----------
    sessions_pokes : list of sessions
        sessions_pokes[sess][trial] = np.array of ports 1..8
    ports_true_all : (n_sessions,)
        True reward port for each session (1..8).
    lag_ports_true_all : (n_sessions,)
        Yesterday's reward port for each session (1..8).
    distance_seq_all : (n_sessions,)
        Distance bins (0..4) per session.
    positionsStartTrial_polar_all : list of arrays
        positionsStartTrial_polar_all[sess] has shape (T,2): [r, theta] at cue.
    Returns
    -------
    session_ll : (n_sessions,)
        Log-likelihood per session.
    mean_ll : float
        Mean log-likelihood over sessions.
    """
    n_sessions = len(sessions_pokes)
    port_angles = np.asarray(port_angles, dtype=float)
    ports_true_all = np.asarray(ports_true_all, dtype=int)
    lag_ports_true_all = np.asarray(lag_ports_true_all, dtype=int)
    distance_seq_all = np.asarray(distance_seq_all, dtype=int)

    session_ll = np.zeros(n_sessions, dtype=float)

    rng = np.random.default_rng(123)

    for i in range(n_sessions):
        sess_pokes = sessions_pokes[i]
        distance_bin = int(distance_seq_all[i])

        # Initialize belief for this session (as in MemoryAgent.reset_episode)
        k_yesterday = float(ky_vector[distance_bin])
        w_today = float(w_vector[distance_bin])

        belief0, f_theta0, _ = initialize_belief(
            episode=i,
            ports_true=ports_true_all,
            lag_ports_true=lag_ports_true_all,
            kappa_today=k_today,
            kappa_yesterday=k_yesterday,
            w_today=w_today,
        )

        prior_ports = f_theta0
        last_visit_trials = np.full(8, -1, dtype=int)

        start_polar = np.asarray(positionsStartTrial_polar_all[i], dtype=float)

        ll = 0.0

        for t, trial_pokes in enumerate(sess_pokes):
            if t >= start_polar.shape[0]:
                break  # no position info for further trials

            r_cur = float(start_polar[t, 0])
            theta_cur = float(start_polar[t, 1])

            for s, port in enumerate(trial_pokes):
                a_idx = int(port) - 1  # convert 1..8 -> 0..7

                probs = heuristic_action_probs(
                    prior_ports=prior_ports,
                    last_visit_trials=last_visit_trials,
                    trial_index=t,
                    step_in_trial=s,
                    r_cur=r_cur,
                    theta_cur=theta_cur,
                    port_angles=port_angles,
                    lambda_dist=lambda_dist,
                    eps_first=eps_first,
                    eps_rest=eps_rest,
                    rng=rng,
                )

                p_a = probs[a_idx]
                if p_a <= 0:
                    ll += -1e9  # avoid -inf, but mark impossible under policy
                else:
                    ll += np.log(p_a)

                # Update internal state as if the agent had chosen this action
                last_visit_trials[a_idx] = t
                r_cur = 1.0
                theta_cur = port_angles[a_idx]

        session_ll[i] = ll

    mean_ll = float(np.mean(session_ll))
    return session_ll, mean_ll


# ---------------------------------------------------------------------- #
# 2) LL for DQN policy                                                   #
# ---------------------------------------------------------------------- #

def sequence_loglik_dqn(
    sessions_pokes: List[List[np.ndarray]],
    ports_true_all: np.ndarray,
    lag_ports_true_all: np.ndarray,
    distance_seq_all: np.ndarray,
    positionsStartTrial_polar_all: List[np.ndarray],
    port_angles: np.ndarray,
    dqn_agent: DQNRecurrentAgent,
    k_today: float,
    ky_vector: List[float],
    w_vector: List[float],
    lambda_dist: float,
    eps_eval: float = 0.05,
) -> Tuple[np.ndarray, float]:
    """
    Compute log-likelihood of REAL sequences under a trained DQN policy.

    We interpret the DQN as epsilon-greedy on Q-values with evaluation epsilon
    `eps_eval` (separate from training epsilon schedule).

    Parameters
    ----------
    sessions_pokes : list of sessions
        sessions_pokes[sess][trial] = np.array of ports 1..8
    dqn_agent : trained DQNRecurrentAgent
        We will use its Q-network to compute Q(s), then construct a
        epsilon-greedy policy with eps_eval.
    eps_eval : float
        Evaluation epsilon (0 = deterministic greedy).
    """
    n_sessions = len(sessions_pokes)
    port_angles = np.asarray(port_angles, dtype=float)
    ports_true_all = np.asarray(ports_true_all, dtype=int)
    lag_ports_true_all = np.asarray(lag_ports_true_all, dtype=int)
    distance_seq_all = np.asarray(distance_seq_all, dtype=int)

    from pomdp.agents.dqn_recurrent import Device  # optional convenience

    device = Device
    session_ll = np.zeros(n_sessions, dtype=float)

    for i in range(n_sessions):
        sess_pokes = sessions_pokes[i]
        distance_bin = int(distance_seq_all[i])

        # Initialize belief/prior for this session (like MemoryAgent.reset_episode)
        k_yesterday = float(ky_vector[distance_bin])
        w_today = float(w_vector[distance_bin])

        belief0, f_theta0, _ = initialize_belief(
            episode=i,
            ports_true=ports_true_all,
            lag_ports_true=lag_ports_true_all,
            kappa_today=k_today,
            kappa_yesterday=k_yesterday,
            w_today=w_today,
        )
        prior_ports = f_theta0
        last_visit_trials = np.full(8, -1, dtype=int)

        start_polar = np.asarray(positionsStartTrial_polar_all[i], dtype=float)

        # hidden state for DQN (None -> zeros internally)
        h = None
        ll = 0.0

        for t, trial_pokes in enumerate(sess_pokes):
            if t >= start_polar.shape[0]:
                break

            r_cur = float(start_polar[t, 0])
            theta_cur = float(start_polar[t, 1])

            for s, port in enumerate(trial_pokes):
                a_idx = int(port) - 1  # 0..7

                # build obs as in MemoryAgent.build_obs
                age = (t - last_visit_trials).astype(float)
                age[age < 0] = 0.0
                age_norm = age / 20.0

                f_t = update_belief(
                    prior_ports=prior_ports,
                    last_visit_trials=last_visit_trials,
                    trial_index=t,
                    reward=None,
                )

                trial_norm = np.array([t / 20.0], dtype=float)
                dist_norm = np.array([distance_bin / 4.0], dtype=float)
                obs = np.concatenate([f_t, age_norm, trial_norm, dist_norm], axis=0)

                # get Q-values for this (s)
                o_torch = torch.as_tensor(obs, dtype=torch.float32, device=device)[None, None, :]
                dqn_agent.q_net.eval()
                with torch.no_grad():
                    q, h = dqn_agent.q_net(o_torch, h)
                    q_last = q[:, -1, :]  # (1, n_actions)
                    q_np = q_last.squeeze(0).cpu().numpy()

                n_ports = q_np.shape[0]
                mx = q_np.max()
                ties = np.flatnonzero(np.isclose(q_np, mx, atol=1e-12))

                probs = np.ones(n_ports, dtype=float) * (eps_eval / n_ports)
                if len(ties) > 0:
                    probs[ties] += (1.0 - eps_eval) / len(ties)

                p_a = probs[a_idx]
                if p_a <= 0:
                    ll += -1e9
                else:
                    ll += np.log(p_a)

                # update belief state with REAL action
                last_visit_trials[a_idx] = t
                r_cur = 1.0
                theta_cur = port_angles[a_idx]

        session_ll[i] = ll

    mean_ll = float(np.mean(session_ll))
    return session_ll, mean_ll


# ---------------------------------------------------------------------- #
# 3) LL for GAIL policy                                                 #
# ---------------------------------------------------------------------- #

def sequence_loglik_gail(
    sessions_pokes: List[List[np.ndarray]],
    ports_true_all: np.ndarray,
    lag_ports_true_all: np.ndarray,
    distance_seq_all: np.ndarray,
    positionsStartTrial_polar_all: List[np.ndarray],
    port_angles: np.ndarray,
    gail: GAILRecurrent,
    k_today: float,
    ky_vector: List[float],
    w_vector: List[float],
) -> Tuple[np.ndarray, float]:
    """
    Compute log-likelihood of REAL sequences under a GAIL recurrent policy.

    We treat the policy as a categorical distribution πθ(a|s) given by
    softmax(logits). For each real action, we accumulate log π(a_t | s_t).
    """
    n_sessions = len(sessions_pokes)
    port_angles = np.asarray(port_angles, dtype=float)
    ports_true_all = np.asarray(ports_true_all, dtype=int)
    lag_ports_true_all = np.asarray(lag_ports_true_all, dtype=int)
    distance_seq_all = np.asarray(distance_seq_all, dtype=int)

    device = next(gail.policy.parameters()).device
    session_ll = np.zeros(n_sessions, dtype=float)

    for i in range(n_sessions):
        sess_pokes = sessions_pokes[i]
        distance_bin = int(distance_seq_all[i])

        # Initialize belief/prior like MemoryAgent.reset_episode
        k_yesterday = float(ky_vector[distance_bin])
        w_today = float(w_vector[distance_bin])

        belief0, f_theta0, _ = initialize_belief(
            episode=i,
            ports_true=ports_true_all,
            lag_ports_true=lag_ports_true_all,
            kappa_today=k_today,
            kappa_yesterday=k_yesterday,
            w_today=w_today,
        )
        prior_ports = f_theta0
        last_visit_trials = np.full(8, -1, dtype=int)

        start_polar = np.asarray(positionsStartTrial_polar_all[i], dtype=float)

        h = None
        ll = 0.0

        for t, trial_pokes in enumerate(sess_pokes):
            if t >= start_polar.shape[0]:
                break

            r_cur = float(start_polar[t, 0])
            theta_cur = float(start_polar[t, 1])

            for s, port in enumerate(trial_pokes):
                a_idx = int(port) - 1

                # obs as in MemoryAgent.build_obs
                age = (t - last_visit_trials).astype(float)
                age[age < 0] = 0.0
                age_norm = age / 20.0

                f_t = update_belief(
                    prior_ports=prior_ports,
                    last_visit_trials=last_visit_trials,
                    trial_index=t,
                    reward=None,
                )

                trial_norm = np.array([t / 20.0], dtype=float)
                dist_norm = np.array([distance_bin / 4.0], dtype=float)
                obs = np.concatenate([f_t, age_norm, trial_norm, dist_norm], axis=0)

                obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)[None, None, :]

                gail.policy.eval()
                with torch.no_grad():
                    logits, h = gail.policy(obs_t, h)   # (1,1,n_actions)
                    logits_last = logits[:, -1, :]      # (1,n_actions)
                    probs = torch.softmax(logits_last, dim=-1).squeeze(0)  # (n_actions,)
                    p_a = float(probs[a_idx].item())

                if p_a <= 0:
                    ll += -1e9
                else:
                    ll += np.log(p_a)

                # update belief with REAL action
                last_visit_trials[a_idx] = t
                r_cur = 1.0
                theta_cur = port_angles[a_idx]

        session_ll[i] = ll

    mean_ll = float(np.mean(session_ll))
    return session_ll, mean_ll
