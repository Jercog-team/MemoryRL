import numpy as np

def extract_real_sequences(ALL_portsPoked_REC):
    """
    Extract within-trial poke sequences from real data.

    Each trial may be:
    - None
    - float NaN
    - an array of ports (some entries NaN)

    Only trials with >=2 valid pokes are included.

    Returns
    -------
    sequences : list of list[int]
        Each list is a trial's valid sequence of ports.
    """
    sequences = []

    for animal in ALL_portsPoked_REC:
        for session in animal:
            for trial in session:
                # Skip empty placeholders
                if trial is None or (isinstance(trial, float) and np.isnan(trial)):
                    continue

                arr = np.array(trial).ravel()
                arr = arr[~np.isnan(arr)]

                if len(arr) >= 2:
                    sequences.append(arr.astype(int).tolist())

    return sequences


def extract_sim_sequences(sim_pokes_sessions_pen):
    """
    Extract within-trial poke sequences from simulation output.

    Converts ports from 0..7 to 1..8.

    Returns
    -------
    sequences : list of list[int]
    """
    sequences = []

    for sess in sim_pokes_sessions_pen:
        for trial in sess:
            if trial is None:
                continue

            arr = np.array(trial).ravel()
            if len(arr) >= 2:
                # Convert sim indexing to biological indexing
                sequences.append([int(p) + 1 for p in arr])

    return sequences


def extract_firstpokes_real(ALL_portsPoked_REC, n_ports=8):
    """
    Collect FIRST poke from each trial in each session.

    Returns
    -------
    seqs : list of list[int]
        Each inner list is the sequence of first-pokes in one session.
    """
    seqs = []

    for animal in ALL_portsPoked_REC:
        for session in animal:
            s = []

            for trial in session:
                if trial is None or (isinstance(trial, float) and np.isnan(trial)):
                    continue

                arr = np.array(trial).ravel()
                arr = arr[~np.isnan(arr)]

                if len(arr) > 0:
                    fp = int(arr[0])
                    if 1 <= fp <= n_ports:
                        s.append(fp)

            if len(s) >= 2:  # Only meaningful if >=2 first-pokes
                seqs.append(s)

    return seqs


def extract_firstpokes_sim(sim_pokes_sessions_pen, n_ports=8):
    """
    Extract first pokes from simulated trials (convert 0..7 → 1..8).

    Returns
    -------
    seqs : list of list[int]
    """
    seqs = []

    for sess in sim_pokes_sessions_pen:
        s = []

        for trial in sess:
            if trial is None:
                continue

            arr = np.array(trial).ravel()

            if len(arr) > 0:
                fp = int(arr[0]) + 1  # convert sim index
                if 1 <= fp <= n_ports:
                    s.append(fp)

        if len(s) >= 2:
            seqs.append(s)

    return seqs


def extract_sessions_trials_real(ALL_portsPoked_REC):
    """
    Build sessions of trials (for last→first transitions) from real data.

    Each trial must contain at least one valid poke.

    Returns
    -------
    sessions : list
        Each session is a list of trial lists (ports as int).
    """
    sessions = []

    for animal in ALL_portsPoked_REC:
        for session in animal:
            trials = []

            for trial in session:
                if trial is None or (isinstance(trial, float) and np.isnan(trial)):
                    continue

                arr = np.array(trial).ravel()
                arr = arr[~np.isnan(arr)]

                if len(arr) >= 1:
                    trials.append(arr.astype(int).tolist())

            if len(trials) >= 2:  # need at least 2 trials to compute transitions
                sessions.append(trials)

    return sessions


def extract_sessions_trials_sim(sim_pokes_sessions_pen):
    """
    Build sessions of trials from simulated data (convert 0..7 → 1..8).

    Returns
    -------
    sessions : list
        List of sessions, each a list of trials.
    """
    sessions = []

    for sess in sim_pokes_sessions_pen:
        trials = []

        for trial in sess:
            if trial is None:
                continue

            arr = np.array(trial).ravel()

            if len(arr) >= 1:
                trials.append([int(p) + 1 for p in arr])

        if len(trials) >= 2:
            sessions.append(trials)

    return sessions




