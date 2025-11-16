# angles.py
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import numpy as np
from constants import ANG_RAD_DICT, N_PORTS, PORTS_ORDER  # adjust import path as needed


def to_polar_mat(xy: np.ndarray) -> np.ndarray:
    """
    Convert a set of 2D Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape (N, 2) where each row is [x, y].

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) where each row is [r, theta]:
        - r: radius (distance from origin)
        - theta: angle in radians in range [-pi, pi]
    """
    r = np.hypot(xy[:, 0], xy[:, 1])
    theta = np.arctan2(xy[:, 1], xy[:, 0])
    return np.column_stack((r, theta))


def convert_to_angular_poked(hist: np.ndarray | Sequence[np.ndarray],
                             by_trial: bool = True) -> np.ndarray:
    """
    Convert port-poke histograms to angular coordinates (radians).

    Ports are mapped to angles using ANG_RAD_DICT.

    Parameters
    ----------
    hist : np.ndarray or sequence of np.ndarray
        If by_trial=True:
            - hist can be:
                * a list/sequence of 2D arrays (n_trials, n_ports),
                * or a single 2D array (n_trials, n_ports).
              In both cases it returns angular matrices with same shape,
              where non-zero entries are replaced by angles and zeros become NaN.
        If by_trial=False:
            - hist is treated as a collection of "sessions" (rows). For each
              element/slice, we:
                * find ports with non-zero counts,
                * build the sequence of their angles,
              and then pad all sequences to the same length with NaNs.

    by_trial : bool, default True
        Whether to preserve the (trial, port) structure (True) or to flatten
        pokes per "session" (False).

    Returns
    -------
    np.ndarray
        If by_trial=True:
            - object array of shape (n_sessions,) where each element is
              a (n_trials, n_ports) float array with angles/NaNs.
        If by_trial=False:
            - 2D array of shape (n_sessions, max_len) with padded sequences
              of angles (NaN padding).
    """
    # ---------- by_trial mode ----------
    if by_trial:
        # Normalize input to a list of 2D arrays called "sessions"
        if isinstance(hist, np.ndarray):
            if hist.ndim == 2:
                hist_sessions: List[np.ndarray] = [hist]
            elif hist.ndim == 3:
                hist_sessions = [hist[i] for i in range(hist.shape[0])]
            else:
                raise ValueError("hist must be 2D or 3D when by_trial=True.")
        else:
            # Assume iterable of 2D arrays (sessions)
            hist_sessions = list(hist)

        ang_ports_sess: List[np.ndarray] = []

        for sess in hist_sessions:
            if sess.ndim != 2:
                raise ValueError("Each session must be a 2D array (n_trials, n_ports).")

            n_trials, n_ports = sess.shape
            ang_ports = np.full((n_trials, n_ports), np.nan, dtype=float)

            for trial_idx in range(n_trials):
                for port_idx in range(n_ports):
                    if sess[trial_idx, port_idx] != 0:
                        # ports are 1-based in ANG_RAD_DICT
                        ang_ports[trial_idx, port_idx] = ANG_RAD_DICT.get(port_idx + 1, np.nan)

            ang_ports_sess.append(ang_ports)

        return np.array(ang_ports_sess, dtype=object)

    # ---------- by_session / flattened mode ----------
    # Here we keep the original semantics: iterate over index, then
    # treat each hist[sess] as one "session" (can be 1D or 2D).
    if isinstance(hist, np.ndarray):
        # If 2D: treat each row as a "session"
        # If 3D: treat each slice hist[sess] as a "session"
        hist_array = hist
    else:
        hist_array = np.array(hist, dtype=float)

    ang_ports_sess: List[List[float]] = []

    for sess_idx in range(len(hist_array)):
        sess_data = hist_array[sess_idx]
        ang_ports: List[float] = []

        # If sess_data is 1D: single vector of port counts
        # If 2D: multiple rows (e.g. trials) for that session
        if sess_data.ndim == 1:
            rows = [sess_data]
        else:
            rows = sess_data

        for row in rows:
            positions = np.where(row != 0)[0]
            if positions.size > 0:
                ang_ports.extend([ANG_RAD_DICT.get(pos + 1, np.nan) for pos in positions])

        ang_ports_sess.append(ang_ports)

    max_length = max(len(lst) for lst in ang_ports_sess) if ang_ports_sess else 0
    ang_ports_sess_padded = np.array(
        [lst + [np.nan] * (max_length - len(lst)) for lst in ang_ports_sess],
        dtype=float,
    )

    return ang_ports_sess_padded


def circular_distance(angle1: np.ndarray | float,
                      angle2: np.ndarray | float) -> np.ndarray | float:
    """
    Compute the circular distance between two angles (or arrays of angles).

    Parameters
    ----------
    angle1 : float or np.ndarray
        First angle(s) in radians.
    angle2 : float or np.ndarray
        Second angle(s) in radians.

    Returns
    -------
    np.ndarray or float
        Minimal distance between angle1 and angle2 on the circle,
        in radians, in [0, pi].
    """
    angle_difference = np.abs(np.array(angle2) - np.array(angle1))
    wrapped_angle_difference = np.minimum(angle_difference, 2 * np.pi - angle_difference)
    return wrapped_angle_difference
