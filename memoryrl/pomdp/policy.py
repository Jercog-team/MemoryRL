# pomdp/policy.py
import numpy as np

from .belief import update_belief


def compute_port_scores(
    prior_ports,
    last_visit_trials,
    trial_index,
    r_cur,
    theta_cur,
    port_angles,
    lambda_dist,
):
    """
    Compute action scores over ports using your current heuristic:

      1) f_t(r) = normalize( f_0(r) * age_t(r) )
      2) score(r) = f_t(r) - lambda_dist * penalty(r)

    with penalty(r) = -2 * r_cur * cos(phi_r - theta_cur).

    Parameters
    ----------
    prior_ports : ndarray, shape (n_ports,)
        Static prior f_0(r) from initialize_belief.
    last_visit_trials : ndarray, shape (n_ports,)
        Last trial index per port; -1 if never.
    trial_index : int
        Current trial index t (0-based).
    r_cur : float
        Current radius in normalized coordinates.
    theta_cur : float
        Current angle in radians.
    port_angles : ndarray, shape (n_ports,)
        Angular position φ_r of each port.
    lambda_dist : float
        Weight for the distance penalty.

    Returns
    -------
    scores : ndarray, shape (n_ports,)
        Action scores.
    """
    port_angles = np.asarray(port_angles, dtype=float)

    # 1) belief over ports at trial t = your age×prior heuristic
    f_t = update_belief(
        prior_ports=prior_ports,
        last_visit_trials=last_visit_trials,
        trial_index=trial_index,
        reward=None,  # currently ignored
    )

    # 2) penalty_j(t,s) = -2 r_cur cos(phi_j - theta_cur)
    penalty_vec = -2.0 * r_cur * np.cos(port_angles - theta_cur)

    scores = f_t - lambda_dist * penalty_vec
    return scores


def epsilon_greedy_policy(
    prior_ports,
    last_visit_trials,
    trial_index,
    r_cur,
    theta_cur,
    port_angles,
    lambda_dist,
    eps_first,
    eps_rest,
    step_in_trial,
    rng,
    atol=1e-12,
):
    """
    Epsilon-greedy policy on top of compute_port_scores.

    Parameters
    ----------
    prior_ports : ndarray, shape (n_ports,)
        Static prior f_0(r).
    last_visit_trials : ndarray, shape (n_ports,)
        Last trial index per port; -1 if never.
    trial_index : int
        Current trial index t.
    r_cur, theta_cur : float
        Current polar coordinates of the animal.
    port_angles : ndarray, shape (n_ports,)
        Port directions φ_r.
    lambda_dist : float
        Distance penalty weight.
    eps_first, eps_rest : float
        Epsilon parameters for first poke vs later pokes.
    step_in_trial : int
        0 for first poke, 1,2,... for later pokes.
    rng : np.random.Generator
        RNG for reproducible exploration.
    atol : float
        Tolerance for argmax ties.

    Returns
    -------
    action : int
        Selected port index in 0..n_ports-1.
    """
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
    if rng.random() < eps:
        # exploration
        return int(rng.integers(0, n_ports))

    # exploitation with random tie-breaking
    mx = np.max(scores)
    ties = np.flatnonzero(np.isclose(scores, mx, atol=atol))
    return int(rng.choice(ties))
