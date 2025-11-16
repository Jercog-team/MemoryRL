# histogram.py

import numpy as np
from collections import Counter
from constants import ANG_RAD_DICT
from utils import circular_distance


def hist_all(port_seq: Sequence[np.ndarray],
             ports_poked: Sequence,
             num_trials: int) -> np.ndarray:
    """
    Build per-trial histograms of pokes for all animals and sessions.

    Parameters
    ----------
    port_seq : sequence of np.ndarray
        Port sequences per animal/session. (Kept for shape consistency; not
        used explicitly, but useful if you want to add checks.)
    ports_poked : nested sequence
        ports_poked[animal, session, trial] is a list/array of port indices
        poked in that trial.
    num_trials : int
        Number of trials per session (loop upper bound; padding length).

    Returns
    -------
    np.ndarray
        hist_seq_arr with shape (n_animals, n_sessions, num_trials, N_PORTS).
        Each entry is the histogram of pokes per port for that trial.
        If no pokes occurred in a trial, the histogram is all zeros.
    """
    all_ports_poked_arr = np.array(ports_poked, dtype=object)

    hist_seq: List[np.ndarray] = []

    # Loop over animals
    for aa in range(len(all_ports_poked_arr)):
        hist_anim: List[np.ndarray] = []

        # Loop over sessions for this animal
        for s, sess in enumerate(all_ports_poked_arr[aa]):
            hist_sess: List[np.ndarray] = []

            # Loop over trials (up to num_trials, padded with zeros if needed)
            for trl in range(num_trials):
                trial_pokes = all_ports_poked_arr[aa, s, trl]

                # If the trial has no pokes (empty or scalar), record zeros
                if np.shape(trial_pokes) == () or np.shape(trial_pokes) == (0,):
                    hist_sess.append(np.zeros(N_PORTS, dtype=int))
                else:
                    # Flatten any nested structure of pokes
                    filtered_pokes = np.array(np.hstack(trial_pokes).tolist())
                    port_counts = Counter(filtered_pokes)

                    # Build vector of counts in PORTS_ORDER (1..8)
                    hist_sess.append(
                        np.array([port_counts[port] for port in PORTS_ORDER], dtype=int)
                    )

            hist_anim.append(np.array(hist_sess, dtype=int))

        hist_seq.append(np.array(hist_anim, dtype=int))

    hist_seq_arr = np.array(hist_seq, dtype=int)
    return hist_seq_arr
