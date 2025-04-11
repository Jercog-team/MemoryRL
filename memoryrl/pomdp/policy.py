import numpy as np


def greedy_policy_marginal_belief(b):

    """
    Selects an action based on the marginal belief over ports.

    Parameters:
        b (np.array): Joint belief over (reward port, tau), shape = [n_ports, n_trials].

    Returns:
        tuple:
            - action (int): Selected port (0-based index) with the highest marginal belief.
            - marginal_belief (np.array): Marginal belief over ports, shape = [n_ports].
    """
    # Sum over the second dimension (tau) to get marginal belief over ports
    marginal_belief = b.sum(axis=1)
    return np.argmax(marginal_belief), marginal_belief


def greedy_policy_last_visit(f_theta, last_visit, t):

    """
    Selects an action based on the last visit times and a weighting factor.

    Parameters:
        f_theta (np.array): Weighting factor for each port, shape = [n_ports].
        last_visit (np.array): Array of last visit times for each port, shape = [n_ports].
        t (int): Current trial index.

    Returns:
        tuple:
            - action (int): Selected port (0-based index) with the highest score.
            - scores (np.array): Scores for each port, shape = [n_ports].
    """

    # Calculate scores based on the last visit times
    scores = f_theta * (t - last_visit)
    return np.argmax(scores), scores


def bellman_policy(state, b_new, last_visit, current_trial, n_trials=20, gamma=0.9, depth=2):
    """
    Bellman-based policy using value iteration with limited depth.

    Parameters:
        state (list): Sequence of past actions (0-based indices).
        b_new (np.array): Joint belief over (reward port, tau), shape = [n_ports, n_trials].
        last_visit (dict): Last trial each port was visited.
        current_trial (int): Current trial index (t).
        n_trials (int): Total number of trials (T).
        gamma (float): Discount factor.
        depth (int): Depth of recursion.

    Returns:
        action (int): Selected port (0-based).
    """
    marginal_belief = np.sum(b_new, axis=1)

    def termination_probability(p):
        lv = last_visit.get(p, -1)
        if lv >= n_trials:
            return 0
        numerator = current_trial - lv
        denominator = n_trials - lv
        return marginal_belief[p] * (numerator / denominator) if denominator > 0 else 0

    def Q(state, action, last_visit, depth_left):
        T_p = termination_probability(action)
        if depth_left == 0 or T_p >= 1.0:
            return T_p
        # Hypothetical future last_visit update
        new_last_visit = last_visit.copy()
        new_last_visit[action] = current_trial
        values = [Q(state + [action], a, new_last_visit, depth_left - 1) for a in range(len(marginal_belief))]
        return T_p + gamma * (1 - T_p) * max(values)

    q_values = [Q(state, a, last_visit, depth) for a in range(len(marginal_belief))]
    return int(np.argmax(q_values)), q_values
