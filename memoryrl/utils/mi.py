import numpy as np
from utils.constants import ANG_RAD_DICT, PORTS_ORDER, N_PORTS


def MemoryIndexbyTrl(hist_seq, port_seq):
    """
    Compute a 'memory index' for a single trial given its poke histogram
    and the correct/relevant port.

    Idea:
    - Ports 1..N_PORTS are placed on the circle with angles given by ANG_RAD_DICT.
    - We take cos(angle) for each port (preference along one axis).
    - We rotate that cosine vector so that the chosen port becomes the
      reference direction.
    - We normalize the histogram and compute the weighted sum of cosines.

    Parameters
    ----------
    hist_seq : array-like, shape (N_PORTS,)
        Histogram of pokes over the ports for one trial.
    port_seq : int
        The current (correct) port index, in 1..N_PORTS.

    Returns
    -------
    MITrl : float
        Memory index for this trial. Positive values → pokes biased toward
        the reference port; negative values → biased away.
        Returns NaN if port_seq <= 0 or hist_seq has no mass.
    """
    # Cosines of the angles associated with each port, in the canonical order
    cosAng = np.cos([ANG_RAD_DICT[p] for p in PORTS_ORDER])

    # Default to NaN in case input is invalid
    MITrl = np.nan

    if port_seq > 0:
        hist_seq = np.asarray(hist_seq, dtype=float)
        total = np.nansum(hist_seq)

        if total > 0:
            # Normalize histogram so that it sums to 1
            Projected_hist_seq_norm = hist_seq / total

            # Rotate cosAng so that `port_seq` becomes the reference direction.
            # PORTS_ORDER is 1..N_PORTS, so `port_seq`-1 gives its index.
            # Using the same shift convention as your original code: (port_seq - 2)
            shift = int(port_seq) - 2
            rotated_cos = np.roll(cosAng, shift)

            # Elementwise product: projected mass along the reference axis
            Projected_hist_seq = rotated_cos * Projected_hist_seq_norm

            # Memory index = sum of projected probabilities
            MITrl = np.nansum(Projected_hist_seq)

    return MITrl


def MemoryIndex_histogram(Big_Hist_data, port_seq, lag_port_seq):
    """
    Compute memory indices from pooled histograms, aligned by current and lagged port.

    For each trial:
    - Circularly shift the 1D histogram (length N_PORTS) so that the
      *current* port (or *lagged* port) is aligned to a reference bin
      (the last port in PORTS_ORDER, i.e. port N_PORTS).
    - Sum shifted histograms across all trials → pooled histograms h, hLags.
    - Compute MemoryIndexbyTrl on each pooled histogram, using `port_seq = N_PORTS`
      as the reference port (the aligned target).

    Parameters
    ----------
    Big_Hist_data : array-like, shape (n_trials, N_PORTS)
        Trial-wise histograms of pokes over ports.
    port_seq : array-like, shape (n_trials,)
        Current port for each trial (1..N_PORTS).
    lag_port_seq : array-like, shape (n_trials,)
        Lagged port for each trial (1..N_PORTS).

    Returns
    -------
    MI_distances : float
        Memory index using histograms aligned by current port.
    MILags_distances : float
        Memory index using histograms aligned by lagged port.
    """
    Big_Hist_data = np.asarray(Big_Hist_data)
    port_seq = np.asarray(port_seq)
    lag_port_seq = np.asarray(lag_port_seq)

    # Align histograms by the current port:
    #   for each trial ss, shift so that `port_seq[ss]` is mapped to the
    #   reference port (N_PORTS).
    h = np.sum(
        [
            np.roll(Big_Hist_data[ss], N_PORTS - int(port_seq[ss]), axis=0)
            for ss in range(len(Big_Hist_data))
        ],
        axis=0,
    )

    # Align histograms by the lagged port
    hLags = np.sum(
        [
            np.roll(Big_Hist_data[ss], N_PORTS - int(lag_port_seq[ss]), axis=0)
            for ss in range(len(Big_Hist_data))
        ],
        axis=0,
    )

    # Compute memory index assuming the aligned port is N_PORTS
    MI_distances = MemoryIndexbyTrl(h, N_PORTS)
    MILags_distances = MemoryIndexbyTrl(hLags, N_PORTS)

    return MI_distances, MILags_distances



def compute_mi_by_distance(big_hist, port_seq, lag_port_seq, distance_seq, n_buckets=5):
    """
    Compute MI for each distance bucket (0,1,2,...).

    Parameters
    ----------
    big_hist : array
        Total pokes per trial (or histogram features).
    port_seq : array
        Current ports.
    lag_port_seq : array
        Previous day's ports.
    distance_seq : array
        Circular distance between ports (in port units).
    n_buckets : int
        Maximum distance + 1.

    Returns
    -------
    MI_2h : ndarray (n_buckets,)
        Memory index using 2h lag (your custom function).
    MI_24h : ndarray (n_buckets,)
        Memory index using 24h lag.
    """
    big_hist = np.asarray(big_hist)
    port_seq = np.asarray(port_seq)
    lag_port_seq = np.asarray(lag_port_seq)
    distance_seq = np.asarray(distance_seq)

    MI_2h = np.full(n_buckets, np.nan)
    MI_24h = np.full(n_buckets, np.nan)

    for d in range(n_buckets):
        mask = (distance_seq == d)
        if not np.any(mask):
            continue

        H_block = big_hist[mask]
        ports_block = port_seq[mask]
        lags_block = lag_port_seq[mask]

        
        MI_2h[d], MI_24h[d] = MemoryIndex_histogram(
            H_block, ports_block, lags_block
        )

    return MI_2h, MI_24h

def ci_from_surrogates(surr, upper=99, lower=1):
    """
    Compute percentile confidence intervals from surrogate MI datasets.

    Parameters
    ----------
    surr : array (n_surrogates, n_buckets)
    upper : int
        Upper percentile (default 99).
    lower : int
        Lower percentile (default 1).

    Returns
    -------
    dict with:
        'low'  : lower percentile values
        'high' : upper percentile values
        'mean' : mean over surrogates
    """
    surr = np.asarray(surr)

    low = np.percentile(surr, lower, axis=0)
    high = np.percentile(surr, upper, axis=0)
    mean = np.nanmean(surr, axis=0)

    return {"low": low, "high": high, "mean": mean}



