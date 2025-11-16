# utils/position.py
from pathlib import Path

import numpy as np

from utils.dataset import dataset_MI
from utils.angles import to_polar_mat


def load_recall_dataset(npz_path: str | Path = "recall_dataset_CONTROL.npz"):
    """
    Load recall dataset (CONTROL) from an .npz file.

    Parameters
    ----------
    npz_path : str or Path
        Path to the .npz file.

    Returns
    -------
    data : dict
        Dictionary with keys:
        'portsPoked_C', 'PORTS_C', 'PORTS_LAGS_C', 'PORTS_2DAYS_C',
        'CueTimes', 'initial_cues', 'finalCues', 'animL_traj',
        'startTimes', 'animal_position', 'boxMeassures', 'animal_ranges'.
    """
    npz_path = Path(npz_path)
    datad = np.load(npz_path, allow_pickle=True)

    data = {
        "portsPoked_C": datad["portsPoked_C"],
        "PORTS_C": datad["PORTS_C"],
        "PORTS_LAGS_C": datad["PORTS_LAGS_C"],
        "PORTS_2DAYS_C": datad["PORTS_2DAYS_C"],
        "CueTimes": datad["CueTimes"],
        "initial_cues": datad["initial_cues"],
        "finalCues": datad["finalCues"],
        "animL_traj": datad["animL_traj"],
        "startTimes": datad["startTimes"],
        "animal_position": datad["animal_position"],
        "boxMeassures": datad["boxMeassures"],
        "animal_ranges": datad["animal_ranges"],
    }
    return data


def build_recall_MI_and_geometry(portsPoked_C, PORTS_C, PORTS_LAGS_C, boxMeassures):
    """
    Run dataset_MI to get mask_nan and build arena geometry per (animal, session).

    Returns
    -------
    Big_Hist_data, CorrectPort_seq, YesCorrectPort_seq, distance_seq,
    arr_all_trials_pokes, arr_all_trials_pokes_polar, arr_polar_pokes,
    mask_nan, boxMeassures_all, port_coords
    """
    # Mutual-information dataset (you already had this function)
    (Big_Hist_data,
     CorrectPort_seq,
     YesCorrectPort_seq,
     distance_seq,
     arr_all_trials_pokes,
     arr_all_trials_pokes_polar,
     arr_polar_pokes,
     mask_nan) = dataset_MI(portsPoked_C, PORTS_C, PORTS_LAGS_C)

    # Flatten box measures across animals/sessions and apply mask_nan
    # each row: [XCenter, YCenter, Radius]
    boxMeassures_all = np.concatenate(np.array(boxMeassures))[mask_nan]

    # Fixed port angles (counter-clockwise, matching ANG_RAD_DICT)
    port_angles = [
        np.pi/4, 0, -np.pi/4, -np.pi/2,
        -3*np.pi/4, np.pi, 3*np.pi/4, np.pi/2
    ]

    # Compute port coordinates per session
    # port_coords[s][i] = (x_i, y_i) for port i in session s
    port_coords = [
        [
            (
                boxMeassures_all[ss][0] + boxMeassures_all[ss][-1] * np.cos(angle),
                boxMeassures_all[ss][1] + boxMeassures_all[ss][-1] * np.sin(angle),
            )
            for angle in port_angles
        ]
        for ss in range(len(boxMeassures_all))
    ]

    return (Big_Hist_data, CorrectPort_seq, YesCorrectPort_seq, distance_seq,
            arr_all_trials_pokes, arr_all_trials_pokes_polar, arr_polar_pokes,
            mask_nan, boxMeassures_all, port_coords)



def normalize_trajectories(animL_traj, animal_ranges, mask_nan):
    """
    Normalize trajectories to a unit circle per session, using animal_ranges.

    Parameters
    ----------
    animL_traj : array-like
        Nested list/array of trajectories per animal/session, each element is
        an array of shape (T, 3): [x, y, t].
    animal_ranges : array-like
        Nested ranges per session: [ [x_min, x_max], [y_min, y_max] ].
    mask_nan : array-like (boolean)
        Mask from dataset_MI selecting valid (animal, session) entries.

    Returns
    -------
    normalized_traj : list of ndarray
        Each element is an array (T, 3) with [x_norm, y_norm, t] where
        x_norm, y_norm are mapped into a unit disk.
    animals_traj_filtered : list of ndarray
        Original (x, y, t) trajectories filtered by mask_nan.
    box_ranges_masked : ndarray
        Corresponding ranges used for normalization (after masking).
    """
    # Flatten by (animal, session) and apply the same mask as for MI
    animals_traj = np.concatenate(animL_traj)[mask_nan]
    box_ranges = np.concatenate(animal_ranges)
    box_ranges_masked = box_ranges[mask_nan]

    normalized_traj = []

    for ss in range(len(animals_traj)):
        traj = np.asarray(animals_traj[ss])
        if traj.ndim != 2 or traj.shape[1] < 3:
            normalized_traj.append(np.full((0, 3), np.nan))
            continue

        x = traj[:, 0]
        y = traj[:, 1]

        # ranges: [[x_min, x_max], [y_min, y_max]]
        x_min, x_max = box_ranges_masked[ss][0]
        y_min, y_max = box_ranges_masked[ss][1]

        # Map x,y into [-1, 1]×[-1, 1]
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        y_norm = 2 * (-y - y_min) / (y_max - y_min) - 1  # note the minus sign (flip)

        # Enforce unit circle: if radius > 1, project back onto circle
        r = np.sqrt(x_norm**2 + y_norm**2)
        outside = r > 1
        x_norm[outside] = x_norm[outside] / r[outside]
        y_norm[outside] = y_norm[outside] / r[outside]

        # Keep original time in the 3rd column
        t = traj[:, 2]
        normalized_traj.append(np.column_stack((x_norm, y_norm, t)))

    return normalized_traj, animals_traj, box_ranges_masked



def compute_adjusted_cue_times(CueTimes, startTimes, mask_nan):
    """
    Adjust cue times by session startTimes and flatten, then apply mask_nan.

    For each animal/session:
        if startTimes[animal][session] > 0:
            adjusted_cues = startTimes[animal][session] + CueTimes[animal][session]
        else:
            adjusted_cues = CueTimes[animal][session]

    Parameters
    ----------
    CueTimes : array-like
        Nested [animal][session] arrays of cue times.
    startTimes : array-like
        Nested [animal][session] scalars with session start times.
    mask_nan : array-like (boolean)
        Same mask used to select valid sessions.

    Returns
    -------
    CueT_flat_masked : ndarray
        1D array of adjusted cue times, flattened across (animal, session)
        and filtered by mask_nan.
    """
    startTrial = np.asarray(CueTimes, dtype=object).copy()

    for animal in range(len(startTimes)):
        for session in range(len(startTimes[animal])):
            if startTimes[animal][session] > 0:
                startTrial[animal][session] = (
                    startTimes[animal][session] + CueTimes[animal][session]
                )

    CueT_flat = np.concatenate(np.array(startTrial))
    CueT_flat_masked = CueT_flat[mask_nan]

    return CueT_flat_masked


def compute_positions_start_trial(normalized_traj, CueT_flat):
    """
    For each session, find the (x,y) position closest in time to each cue time.

    Parameters
    ----------
    normalized_traj : list of ndarray
        Each element is a (T, 3) array: [x_norm, y_norm, t].
    CueT_flat : ndarray
        Flattened cue times aligned with the same session indexing as
        normalized_traj (the mask_nan alignment must already be done).

    Returns
    -------
    positionsStartTrial : list of ndarray or None
        positionsStartTrial[s] is either:
            - an array of shape (n_cues_in_session, 2) with [x, y] positions
              at cue onset
            - or None if session has no valid trajectory or no cue times.
    """
    positionsStartTrial = [None] * len(normalized_traj)

    # Here we assume CueT_flat is already indexed session-wise.
    # If your CueT is session-wise array instead of flat, adapt indexing.
    CueT = np.asarray(CueT_flat, dtype=object)

    for sess in range(len(normalized_traj)):
        traj = np.asarray(normalized_traj[sess])

        # Expecting (N, 3): [x, y, t]
        if traj.ndim != 2 or traj.shape[1] < 3 or np.all(np.isnan(traj)):
            continue  # skip invalid sessions

        xy = traj[:, :2]                 # (N, 2)
        t_traj = traj[:, 2].astype(float)

        # Replace NaN times with inf to avoid being chosen as nearest
        t_traj = np.where(np.isnan(t_traj), np.inf, t_traj)

        # This assumes CueT is session-wise; if it's fully flattened,
        # you'll likely pass in an array already indexed by sess.
        cue_times = np.asarray(CueT[sess]).astype(float)
        cue_times = cue_times[~np.isnan(cue_times)]
        if cue_times.size == 0:
            continue

        # Compute |t_traj - cue_times| for all pairs (traj point, cue)
        diff = np.abs(t_traj[:, None] - cue_times[None, :])
        closest_idx = np.argmin(diff, axis=0)   # for each cue

        CueTimes_onset = xy[closest_idx, :]     # (N_cues, 2)
        positionsStartTrial[sess] = CueTimes_onset

    return positionsStartTrial


def positions_to_polar(positionsStartTrial):
    """
    Convert session-wise start positions from Cartesian to polar coordinates.

    Parameters
    ----------
    positionsStartTrial : list of ndarray or None
        Each element is array (n_cues, 2) or None.

    Returns
    -------
    positionsStartTrial_polar : list of ndarray or None
        Same structure, but each (n_cues, 2) is [r, theta].
    """
    positionsStartTrial_polar = []
    for pos in positionsStartTrial:
        if pos is None:
            positionsStartTrial_polar.append(None)
        else:
            positionsStartTrial_polar.append(to_polar_mat(pos))
    return positionsStartTrial_polar


