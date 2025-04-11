import numpy as np
import pickle
import importlib.resources as pkg_resources
from distributions import __name__ as dist_pkg


def initialize_belief_von_misses(r_true_today,
                                 r_true_yesterday,
                                 kappa=3,
                                 w_today=0.3,
                                 w_yesterday=0.2,
                                 n_ports=8,
                                 n_trials=20,
                                 water_availability_dist=True):

    """
    Initializes the belief state using a von Mises distribution.

    Args:
        kappa (float): Concentration parameter for the von Mises distribution.
        w_today (float): Weight for today's belief.
        w_yesterday (float): Weight for yesterday's belief.
        water_availability_dist (array): Distribution of water availability.
        n_ports (int): Number of ports.
        n_trials (int): Number of trials.

    Returns:
        tuple: A tuple containing the prior belief state (b_prior), today's true port (r_true_today),
               the true trial (tau_true), last visit times (last_visit), and yesterday's true port (r_true_yesterday).
    """

    thetas = np.linspace(0, 2*np.pi, n_ports+1)[:-1]
    
    
    f_r_today = np.exp(kappa * np.cos(thetas - thetas[r_true_today-1]))
    f_r_today /= f_r_today.sum()  
    
    f_r_yesterday = np.exp(kappa * np.cos(thetas - thetas[r_true_yesterday-1]))
    f_r_yesterday /= f_r_yesterday.sum()


    f_r_exploration = np.exp(0.015 * np.cos(thetas - thetas[r_true_today-1]))
    f_r_exploration /= f_r_exploration.sum()  
    
    f_r = w_today * f_r_today + w_yesterday * f_r_yesterday + 0.001 * f_r_exploration
    f_r /= f_r.sum()
    
    if water_availability_dist:
        with pkg_resources.files(dist_pkg).joinpath("water_aval_dist.pkl").open("rb") as f1:
            water_availability_dist = pickle.load(f1)
        
        # Adjust the distribution to match n_trials
        if len(water_availability_dist) != n_trials:
            x_original = np.linspace(0, 1, len(water_availability_dist))
            x_target = np.linspace(0, 1, n_trials)
            water_availability_dist = np.interp(x_target, x_original, water_availability_dist)
        
        water_availability_dist /= np.sum(water_availability_dist)  # Normalize
        tau_true = np.random.choice(np.arange(1, n_trials + 1), p=water_availability_dist)
        g_tau = water_availability_dist

    else:
        tau_true = np.random.randint(1, n_trials + 1)
        g_tau = np.ones(n_trials) / n_trials
    
    b_prior = np.outer(f_r, g_tau)

    last_visit = np.zeros(n_ports, dtype=int)
    
    return b_prior, tau_true, last_visit, f_r



def initialize_belief_deltas(r_true_today,
                             r_true_yesterday,
                             w_today=0.3,
                             w_yesterday=0.2,
                             n_ports=8,
                             n_trials=20,
                             water_availability_dist=True):
    """
    Initializes the belief state using delta functions for today's and yesterday's ports.

    Args:
        w_today (float): Weight for today's belief.
        w_yesterday (float): Weight for yesterday's belief.
        water_availability_dist (array): Use real distribution of water availability (yes or no).
        n_ports (int): Number of ports.
        n_trials (int): Number of trials.

    Returns:
        tuple: A tuple containing the prior belief state (b_prior), today's true port (r_true_today),
               the true trial (tau_true), last visit times (last_visit), and yesterday's true port (r_true_yesterday).
    """


    # Initialize belief for today's correct port
    f_r_today = np.zeros(n_ports)
    f_r_today[r_true_today-1] = 1.0  # Only the correct port of today has a non-zero value

    # Initialize belief for yesterday's correct port
    f_r_yesterday = np.zeros(n_ports)
    f_r_yesterday[r_true_yesterday-1] = 1.0  # Only the correct port of yesterday has a non-zero value

    # Combine today's and yesterday's beliefs
    f_r = w_today * f_r_today + w_yesterday * f_r_yesterday
    f_r /= f_r.sum()  # Normalize the combined belief

    if water_availability_dist:


        with pkg_resources.files(dist_pkg).joinpath("water_aval_dist.pkl").open("rb") as f1:
            water_availability_dist = pickle.load(f1)
        
        # Adjust the distribution to match n_trials
        if len(water_availability_dist) != n_trials:
            x_original = np.linspace(0, 1, len(water_availability_dist))
            x_target = np.linspace(0, 1, n_trials)
            water_availability_dist = np.interp(x_target, x_original, water_availability_dist)
        
        water_availability_dist /= np.sum(water_availability_dist)  # Normalize
        tau_true = np.random.choice(np.arange(1, n_trials + 1), p=water_availability_dist)
        g_tau = water_availability_dist

    else:
        tau_true = np.random.randint(1, n_trials + 1)
        g_tau = np.ones(n_trials) / n_trials


    # Create the initial belief state
    b_prior = np.outer(f_r, g_tau)

    # Initialize last visit times
    last_visit = np.zeros(n_ports, dtype=int)
    
    return b_prior, tau_true, last_visit, f_r




def update_belief(b, w, p, t, last_visit):
    """
    Updates the belief state based on the observed reward and port.

    Args:
        b (ndarray): Current belief state.
        w (int): Reward indicator (1 if reward is observed, 0 otherwise).
        p (int): Port index where the observation occurred.
        t (int): Current trial number.
        last_visit (ndarray): Array tracking the last visit times for each port.

    Returns:
        tuple: A tuple containing the updated belief state (b_new), termination probability (term_prob),
               and updated last visit times (last_visit).
    """
    n_ports, n_trials = b.shape
    b_new = np.copy(b)
    
    reward_likelihood = (t >= np.arange(n_trials)) * (p == np.arange(n_ports)[:, None])
    
    if w == 1:
        b_new[:, :] = 0
        b_new[p, :] = reward_likelihood[p, :]
    else:
        update_factor = 1 - reward_likelihood
        b_new *= update_factor
    
    b_new /= np.sum(b_new) if np.sum(b_new) > 0 else 1
    
    marginal_belief = np.sum(b_new, axis=1)
    term_prob = marginal_belief[p] * (t - last_visit[p]) / (n_trials - last_visit[p])
    
    last_visit[p] = t
    
    return b_new, term_prob, last_visit