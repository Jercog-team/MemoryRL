import numpy as np
from histogram import MemoryIndex_histogram



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



