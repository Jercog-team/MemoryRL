import numpy as np


def row_normalize_probs(M):
    """
    Row-normalize a matrix of non-negative counts into probabilities.

    Parameters
    ----------
    M : array-like (2D)
        Count matrix where rows represent 'from' categories and
        columns represent 'to' categories.

    Returns
    -------
    P : ndarray
        Row-normalized probability matrix. Rows with sum 0 are left as 0.
    """
    M = M.astype(float)
    rs = M.sum(axis=1, keepdims=True)      # row sums
    rs[rs == 0] = 1.0                       # avoid division by zero
    P = M / rs                              # normalize each row
    return np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)


def transition_counts(sequences, n_ports=8):
    """
    Count within-trial transitions a→b over all sequences.

    Parameters
    ----------
    sequences : list of list-like
        Each sequence is a trial's sequence of pokes (ports 1..n_ports).
    n_ports : int
        Number of ports.

    Returns
    -------
    C : ndarray (n_ports x n_ports)
        C[a-1, b-1] = number of transitions from port a to port b.
    """
    C = np.zeros((n_ports, n_ports), dtype=int)

    for tr in sequences:
        arr = np.array(tr, dtype=int).ravel()
        if arr.size < 2:
            continue
        for a, b in zip(arr[:-1], arr[1:]):
            # Count only valid ports
            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1
    return C


def firstpoke_transition_counts(seqs, n_ports=8):
    """
    Count transitions between FIRST pokes of consecutive trials within a session.

    Parameters
    ----------
    seqs : list of lists
        Each inner list contains the first-poke port of each trial in a session.
    n_ports : int
        Number of ports.

    Returns
    -------
    C : ndarray (n_ports x n_ports)
        Transition counts between first pokes of trials.
    """
    C = np.zeros((n_ports, n_ports), dtype=int)

    for s in seqs:
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1
    return C


def lastfirst_transition_counts(session_trials, n_ports=8):
    """
    Count transitions from LAST poke of trial t → FIRST poke of trial t+1.

    Parameters
    ----------
    session_trials : list
        List of sessions; each session is a list of trials,
        and each trial is a list/array of ports (1..n_ports).
        Each trial must have at least one legitimate (non-NaN) entry.
    n_ports : int

    Returns
    -------
    C : ndarray (n_ports x n_ports)
        Transition counts between last→first across trials.
    """
    C = np.zeros((n_ports, n_ports), dtype=int)

    for sess in session_trials:
        if len(sess) < 2:
            continue

        for t in range(len(sess) - 1):
            a_arr = np.array(sess[t]).ravel()
            b_arr = np.array(sess[t + 1]).ravel()

            # Remove NaN pokes
            a_arr = a_arr[~np.isnan(a_arr)]
            b_arr = b_arr[~np.isnan(b_arr)]
            if a_arr.size == 0 or b_arr.size == 0:
                continue

            a = int(a_arr[-1])  # last poke of trial t
            b = int(b_arr[0])   # first poke of trial t+1

            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1

    return C

