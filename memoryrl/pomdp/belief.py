# pomdp/belief.py
import numpy as np
from scipy.stats import vonmises


def initialize_belief(
    episode,
    ports_true,
    lag_ports_true,
    kappa_today,
    kappa_yesterday,
    w_today=0.5,
):
    """
    Initialize the joint belief P(r, τ) for a given episode.

    The idea:
    - There are 8 possible reward ports placed on a circle.
    - The animal has a noisy memory of today's reward port and yesterday's port,
      each modeled as a von Mises distribution on the circle.
    - The initial prior over ports f_θ(r) is a convex combination of:
        * dist_today  ~ vonMises(centered at today's port, kappa_today)
        * dist_yesterday ~ vonMises(centered at yesterday's port, kappa_yesterday)
      with weight w_today for "today" and (1 - w_today) for "yesterday".
    - The initial prior over trial-of-availability g_ψ(τ) is uniform over 20 trials.
    - The joint belief is P(r, τ) = f_θ(r) * g_ψ(τ).

    Parameters
    ----------
    episode : int
        Index of the current episode (used to index ports_true and lag_ports_true).
    ports_true : array-like of int
        True reward port indices for *today*, shape (n_episodes,).
        Values should be in 1..8.
    lag_ports_true : array-like of int
        True reward port indices for *yesterday*, shape (n_episodes,).
        Values should be in 1..8.
    kappa_today : float
        Concentration parameter κ for today's von Mises (larger = more precise memory).
    kappa_yesterday : float
        Concentration parameter κ for yesterday's von Mises.
    w_today : float, optional (default=0.5)
        Weight in [0, 1] for today's distribution.
        Yesterday's weight is (1 - w_today).

    Returns
    -------
    belief : ndarray, shape (8, 20)
        Initial joint belief P(r, τ) over 8 ports and 20 possible trial indices.
    f_theta : ndarray, shape (8,)
        Marginal prior over ports P(r) after combining today/yesterday memories.
    r_true : int
        Today's true reward port index for this episode.
    """
    # There are 8 discrete ports placed evenly on the circle.
    ports = np.arange(8)  # 0,1,...,7
    angles = ports * 2 * np.pi / 8  # corresponding angular positions

    # Index today's and yesterday's ports for this episode
    r_true = ports_true[episode]          # in 1..8
    yesterday_port = lag_ports_true[episode]  # in 1..8

    # Von Mises distributions centered at the angles of today's and yesterday's ports
    dist_today = vonmises.pdf(
        angles,
        kappa_today,
        loc=angles[r_true - 1]  # r_true is 1-based, angles is 0-based
    )
    dist_yesterday = vonmises.pdf(
        angles,
        kappa_yesterday,
        loc=angles[yesterday_port - 1]
    )

    # Normalize each distribution to sum to 1 (safety against numerical drift)
    dist_today /= dist_today.sum()
    dist_yesterday /= dist_yesterday.sum()

    # Convex combination of today's and yesterday's memories
    f_theta = w_today * dist_today + (1.0 - w_today) * dist_yesterday

    # Final normalization (just to be absolutely sure it sums to 1)
    f_theta /= f_theta.sum()

    # Uniform prior over τ (trial of availability), here 20 possible trials
    g_psi = np.ones(20) / 20.0

    # Joint belief as outer product P(r, τ) = f_θ(r) * g_ψ(τ)
    P_r_tau = np.outer(f_theta, g_psi)

    # Copy so we don't accidentally modify P_r_tau outside
    belief = np.copy(P_r_tau)

    return belief, f_theta, r_true
