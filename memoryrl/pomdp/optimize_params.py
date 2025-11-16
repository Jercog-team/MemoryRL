"""
Parameter optimisation for the POMDP using log-likelihood of
real poke sequences.

The idea:

- We optimise a *memory parameter vector* θ_mem:
    (k_today, ky_vector[5], w_vector[5], eps_first, eps_rest, lambda_dist)

- For each θ_mem, we build a policy using a `policy_factory`:
    policy = policy_factory(params_mem)

- Then we compute the negative average log-likelihood of the
  observed real sequences under that policy, using the same
  environment (geometry, starting positions, etc.)

You can use the same code for:
    - GreedyMemoryPolicy (analytic)
    - DQNRecurrentPolicy (if you define feature_fn & freeze weights)
    - GAILRecurrentPolicy (same idea)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Callable, Tuple

import numpy as np
from scipy.optimize import minimize

from .policies import BasePolicy
from .belief import initialize_belief


@dataclass
class LLData:
    """
    Data needed to compute log-likelihood on REAL sequences.
    All arrays must be in the same canonical session order.
    """
    positionsStartTrial_polar: Sequence[np.ndarray]    # list[session] of (T,2)
    distance_seq: np.ndarray                           # (n_sessions,), 0..4
    correct_ports: np.ndarray                          # (n_sessions,), 1..8
    yesterday_ports: np.ndarray                        # (n_sessions,), 1..8
    real_pokes_sessions_0based: Sequence[Sequence[Sequence[int]]]
    port_angles: np.ndarray                            # (8,)


# -------------------- parameter packing/unpacking --------------------

def _softplus(z: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)


def _inv_softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(x > 20, x, np.log(np.expm1(x)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _inv_sigmoid(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1.0 - 1e-9)
    return np.log(p / (1.0 - p))


def pack_params(
    k_today: float,
    ky_vector: np.ndarray,
    w_vector: np.ndarray,
    eps_first: float,
    eps_rest: float,
    lambda_dist: float,
) -> np.ndarray:
    """
    Pack physical parameters into an unconstrained vector z (R^14)
    using softplus / sigmoid transforms.
    """
    ky_vector = np.asarray(ky_vector, dtype=float)
    w_vector = np.asarray(w_vector, dtype=float)
    if ky_vector.shape[0] != 5 or w_vector.shape[0] != 5:
        raise ValueError("ky_vector and w_vector must have length 5.")

    z = np.zeros(14, dtype=float)
    z[0] = _inv_softplus(np.array(k_today))[0]
    z[1:6] = _inv_softplus(ky_vector)
    z[6:11] = _inv_sigmoid(w_vector)
    z[11] = _inv_sigmoid(np.array(eps_first))[0]
    z[12] = _inv_sigmoid(np.array(eps_rest))[0]
    z[13] = _inv_softplus(np.array(lambda_dist))[0]
    return z


def unpack_params(z: np.ndarray) -> Dict[str, Any]:
    """
    Inverse of pack_params: map unconstrained vector z back to
    physical parameter dictionary.
    """
    z = np.asarray(z, dtype=float)
    if z.shape[0] != 14:
        raise ValueError("z must have length 14 (k_today, ky[5], w[5], eps_first, eps_rest, lambda_dist).")

    k_today = float(_softplus(np.array(z[0]))[0])
    ky_vector = _softplus(z[1:6])
    w_vector = _sigmoid(z[6:11])
    eps_first = float(_sigmoid(np.array(z[11]))[0])
    eps_rest = float(_sigmoid(np.array(z[12]))[0])
    lambda_dist = float(_softplus(np.array(z[13]))[0])

    return dict(
        k_today=k_today,
        ky_vector=ky_vector,
        w_vector=w_vector,
        eps_first=eps_first,
        eps_rest=eps_rest,
        lambda_dist=lambda_dist,
    )


# -------------------- negative log-likelihood ------------------------

def build_session_order_for_ll(distance_seq: Sequence[int], oversample_times: int = 1) -> np.ndarray:
    """
    Same oversampling as in simulation: distance_bin == 0 sessions
    repeated oversample_times times. Used to weight LL by distance.
    """
    distance_seq = np.asarray(distance_seq, dtype=int)
    order = []
    for s in range(len(distance_seq)):
        if int(distance_seq[s]) == 0:
            order.extend([s] * oversample_times)
        else:
            order.append(s)
    return np.array(order, dtype=int)


def negative_avg_log_likelihood(
    z: np.ndarray,
    ll_data: LLData,
    policy_factory: Callable[[Dict[str, Any]], BasePolicy],
    oversample_times: int = 1,
    max_pokes_per_trial: int = 4,
) -> float:
    """
    Objective function for optimisation: negative average log-likelihood
    of real sequences under the model/policy.

    Parameters
    ----------
    z : np.ndarray
        Unconstrained parameter vector (14 dims) produced by pack_params().
    ll_data : LLData
        Real data bundle.
    policy_factory : callable
        Function that maps params_mem (dict) -> policy instance.
        It will be called once per evaluation of z.
    oversample_times : int
        If >1, sessions with distance_bin==0 will be repeated in the
        session_order; this reproduces your weighting trick.
    max_pokes_per_trial : int
        Maximum number of pokes considered per trial when scoring LL.

    Returns
    -------
    float
        Negative average log-likelihood (to be minimised).
    """
    params = unpack_params(z)
    k_today = params["k_today"]
    ky_vector = params["ky_vector"]
    w_vector = params["w_vector"]
    eps_first = params["eps_first"]
    eps_rest = params["eps_rest"]
    lambda_dist = params["lambda_dist"]

    # Build policy instance (GreedyMemoryPolicy / DQNRecurrentPolicy / GAILRecurrentPolicy)
    from .policies import GreedyMemoryPolicy  # ensure no circular import at module load
    if policy_factory is None:
        # fallback: default to greedy if no factory provided
        policy = GreedyMemoryPolicy(
            k_today=k_today,
            ky_vector=ky_vector,
            w_vector=w_vector,
            eps_first=eps_first,
            eps_rest=eps_rest,
            lambda_dist=lambda_dist,
            port_angles=ll_data.port_angles,
        )
    else:
        policy = policy_factory(params)

    # Memory params passed to initialize_belief
    params_mem = dict(
        k_today=k_today,
        ky_vector=ky_vector,
        w_vector=w_vector,
    )

    distance_seq = np.asarray(ll_data.distance_seq, dtype=int)
    correct_ports = np.asarray(ll_data.correct_ports, dtype=int)
    yesterday_ports = np.asarray(ll_data.yesterday_ports, dtype=int)

    session_order = build_session_order_for_ll(distance_seq, oversample_times)

    total_logp = 0.0
    total_count = 0

    for sess_idx in session_order:
        sess_idx = int(sess_idx)
        distance_bin = int(distance_seq[sess_idx])      # 0..4
        r_today = int(correct_ports[sess_idx])          # 1..8
        r_yest = int(yesterday_ports[sess_idx])         # 1..8

        # Build f_theta for this session
        belief_init, f_theta, _ = initialize_belief(
            episode=sess_idx,
            ports_true=correct_ports,
            lag_ports_true=yesterday_ports,
            kappa_today=k_today,
            kappa_yesterday=float(ky_vector[distance_bin]),
            w_today=float(w_vector[distance_bin]),
        )
        f_theta = np.asarray(f_theta, dtype=float)

        T_r = np.full(8, -1, dtype=int)
        start_polar = np.asarray(ll_data.positionsStartTrial_polar[sess_idx], dtype=float)
        real_trials = ll_data.real_pokes_sessions_0based[sess_idx]

        policy.reset_episode()

        T_max = min(len(start_polar), len(real_trials))
        for t in range(T_max):
            r_cur, theta_cur = float(start_polar[t, 0]), float(start_polar[t, 1])
            tr = list(real_trials[t])
            if len(tr) == 0:
                continue
            n_steps = min(len(tr), max_pokes_per_trial)

            for step in range(n_steps):
                a = int(tr[step])
                if not (0 <= a < 8):
                    continue

                state = {
                    "t": t,
                    "step": step,
                    "distance_bin": distance_bin,
                    "T_r": T_r,
                    "f_theta": f_theta,
                    "r_cur": r_cur,
                    "theta_cur": theta_cur,
                    "today_port": r_today - 1,
                    "yesterday_port": r_yest - 1,
                    "tau_true": None,  # reward timing irrelevant for LL
                }
                logp = policy.log_prob_of_action(state, a)
                total_logp += logp
                total_count += 1

                # update environment for next decision
                T_r[a] = t
                r_cur = 1.0
                theta_cur = ll_data.port_angles[a]

    if total_count == 0:
        return 1e9
    return float(-total_logp / total_count)


# ----------------------- convenience wrapper -------------------------

def fit_memory_params_with_nelder_mead(
    init_params: Dict[str, Any],
    ll_data: LLData,
    policy_factory: Callable[[Dict[str, Any]], BasePolicy] = None,
    oversample_times: int = 1,
    max_pokes_per_trial: int = 4,
    maxiter: int = 400,
    tol: float = 1e-3,
) -> Tuple[Dict[str, Any], float]:
    """
    Simple Nelder–Mead optimisation wrapper.

    Parameters
    ----------
    init_params : dict
        Initial guess in physical space:
            {
              "k_today": float,
              "ky_vector": np.array(5),
              "w_vector": np.array(5),
              "eps_first": float,
              "eps_rest": float,
              "lambda_dist": float,
            }
    ll_data : LLData
        Real sequences and geometry.
    policy_factory : callable or None
        If None, default to GreedyMemoryPolicy. Otherwise, use this to
        create the policy at each evaluation.
    oversample_times : int
        Oversampling factor for distance_bin==0 sessions.
    maxiter : int, tol : float
        Nelder–Mead stopping settings.

    Returns
    -------
    best_params : dict
        Best-fit parameters in physical space.
    best_value : float
        Minimum negative average log-likelihood.
    """
    z0 = pack_params(
        k_today=init_params["k_today"],
        ky_vector=np.asarray(init_params["ky_vector"], dtype=float),
        w_vector=np.asarray(init_params["w_vector"], dtype=float),
        eps_first=init_params["eps_first"],
        eps_rest=init_params["eps_rest"],
        lambda_dist=init_params["lambda_dist"],
    )

    def obj(z):
        return negative_avg_log_likelihood(
            z=z,
            ll_data=ll_data,
            policy_factory=policy_factory,
            oversample_times=oversample_times,
            max_pokes_per_trial=max_pokes_per_trial,
        )

    res = minimize(
        obj,
        z0,
        method="Nelder-Mead",
        options=dict(maxiter=maxiter, xatol=tol, fatol=tol, adaptive=True),
    )

    best_params = unpack_params(res.x)
    best_value = float(res.fun)
    return best_params, best_value
