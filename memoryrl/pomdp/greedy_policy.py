import numpy as np


def greedy_policy_marginal_belief(b):
    marginal_belief = b.sum(axis=1)
    return np.argmax(marginal_belief), marginal_belief


def greedy_policy_last_visit(f_theta, last_visit, t):
    scores = f_theta * (t - last_visit)
    return np.argmax(scores), scores