"""
memoryrl.viz.report
===================

Helpers to compare REAL vs SIM metrics:

- F_theta (two-peak von Mises over ports)
- Transition matrices (within-trial, first-poke, last→first)
- Histograms by distance + KL
- Memory Index by distance (2h, 24h)
- Optional surrogate CIs
- Optional PDF report

You can use either:
  - compute_memory_metrics(...)  -> metrics dict, no plotting
  - generate_memory_report(...)  -> full PDF (and also returns metrics)
  - individual plotting helpers  -> interactive figures
"""

import os
from typing import Optional, Tuple, Sequence, Callable, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
from collections import Counter

# -----------------------------
# Optional SciPy fallbacks
# -----------------------------
try:
    from scipy.stats import sem as _scipy_sem, entropy as _scipy_entropy

    def _sem(a, axis=0, nan_policy='omit'):
        return _scipy_sem(a, axis=axis, nan_policy=nan_policy)

    def _kl(p, q):
        return _scipy_entropy(p, q)

except Exception:
    def _sem(a, axis=0, nan_policy='omit'):
        a = np.asanyarray(a, dtype=float)
        if nan_policy == 'omit':
            n = np.sum(~np.isnan(a), axis=axis)
            s = np.nanstd(a, axis=axis, ddof=1)
        else:
            n = a.shape[axis]
            s = np.std(a, axis=axis, ddof=1)
        with np.errstate(invalid='ignore', divide='ignore'):
            return s / np.sqrt(np.maximum(n, 1))

    def _kl(p, q):
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        eps = 1e-12
        p = np.clip(p, eps, np.inf); p /= p.sum() if p.sum() > 0 else 1.0
        q = np.clip(q, eps, np.inf); q /= q.sum() if q.sum() > 0 else 1.0
        return np.sum(p * (np.log(p) - np.log(q)))


# -----------------------------
# F-THETA (two-peak von Mises)
# -----------------------------
def von_mises_pdf(angles: np.ndarray, kappa: float, mu: float) -> np.ndarray:
    num = np.exp(kappa * np.cos(angles - mu))
    den = 2.0 * np.pi * np.i0(kappa)
    return num / den


def compute_f_theta(
    n_ports: int = 8,
    today_port: int = 3,
    yesterday_port: int = 6,
    k_today: float = 4.0,
    k_yesterday: float = 2.0,
    w: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute two-peak von Mises mixture over ports:
      f_theta(r) = w * VM(k_today, today_port) + (1-w)*VM(k_yest, yest_port)
    """
    ports_0b = np.arange(n_ports)          # 0..n-1
    ports_1b = ports_0b + 1                # 1..n
    angles = ports_0b * 2 * np.pi / n_ports

    t_idx = int(today_port) - 1
    y_idx = int(yesterday_port) - 1

    dist_today = von_mises_pdf(angles, k_today, angles[t_idx])
    dist_yest  = von_mises_pdf(angles, k_yesterday, angles[y_idx])

    f_theta = w * dist_today + (1 - w) * dist_yest
    f_theta = np.maximum(f_theta, 0.0)
    if f_theta.sum() > 0:
        f_theta = f_theta / f_theta.sum()
    else:
        f_theta = np.full_like(f_theta, 1.0 / len(f_theta))

    return ports_1b, f_theta


def plot_f_theta(
    ax: plt.Axes,
    ports_1b: np.ndarray,
    f_theta: np.ndarray,
    today_port: int,
    yesterday_port: int,
    title: str = "F_theta (two-peak von Mises)",
):
    ax.bar(ports_1b, f_theta, alpha=0.5, label="Combined Distribution")
    ax.plot([today_port], [f_theta[today_port-1]], "o", label="Today's Port")
    ax.plot([yesterday_port], [f_theta[yesterday_port-1]], "s", label="Yesterday's Port")
    ax.set_xlabel("Port Number")
    ax.set_ylabel("Probability")
    ax.set_xticks(ports_1b)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


# ---------------------------------------------
# Transition matrices utilities (probabilities)
# ---------------------------------------------
def row_normalize_probs(M: np.ndarray) -> np.ndarray:
    M = M.astype(float)
    rs = M.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    P = M / rs
    return np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)


def transition_counts(sequences: Sequence[Sequence[int]], n_ports: int = 8) -> np.ndarray:
    """Within-trial transitions."""
    C = np.zeros((n_ports, n_ports), dtype=int)
    for tr in sequences:
        arr = np.array(tr, dtype=int).ravel()
        if arr.size < 2:
            continue
        for a, b in zip(arr[:-1], arr[1:]):
            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1
    return C


def firstpoke_transition_counts(seqs: Sequence[Sequence[int]], n_ports: int = 8) -> np.ndarray:
    """Trial-to-trial transitions of FIRST poke."""
    C = np.zeros((n_ports, n_ports), dtype=int)
    for s in seqs:
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1
    return C


def lastfirst_transition_counts(session_trials: Sequence[Sequence[Sequence[int]]], n_ports: int = 8) -> np.ndarray:
    """
    Last→First across trials:
      from last poke of trial t → first poke of trial t+1.
    """
    C = np.zeros((n_ports, n_ports), dtype=int)
    for sess in session_trials:
        if len(sess) < 2:
            continue
        for t in range(len(sess) - 1):
            a_arr = np.array(sess[t]).ravel()
            b_arr = np.array(sess[t + 1]).ravel()
            a_arr = a_arr[~np.isnan(a_arr)]
            b_arr = b_arr[~np.isnan(b_arr)]
            if a_arr.size == 0 or b_arr.size == 0:
                continue
            a = int(a_arr[-1])
            b = int(b_arr[0])
            if 1 <= a <= n_ports and 1 <= b <= n_ports:
                C[a - 1, b - 1] += 1
    return C


def plot_prob(ax: plt.Axes, M: np.ndarray, title: str, vmin: float, vmax: float, cmap: str = "Blues"):
    im = ax.imshow(M, origin="upper", aspect="equal", interpolation="nearest",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("To port")
    ax.set_ylabel("From port")
    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(range(1, M.shape[1] + 1))
    ax.set_yticklabels(range(1, M.shape[0] + 1))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8)
    return im


# ---------------------------------------------
# MI by distance + surrogate stats
# ---------------------------------------------
def compute_mi_by_distance(
    big_hist: np.ndarray,
    port_seq: np.ndarray,
    lag_port_seq: np.ndarray,
    distance_seq: np.ndarray,
    mi_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]],
    n_buckets: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    big_hist = np.asarray(big_hist)
    port_seq = np.asarray(port_seq)
    lag_port_seq = np.asarray(lag_port_seq)
    distance_seq = np.asarray(distance_seq)

    MI_2h  = np.full(n_buckets, np.nan)
    MI_24h = np.full(n_buckets, np.nan)

    for d in range(n_buckets):
        mask = (distance_seq == d)
        if not np.any(mask):
            continue
        H_block     = big_hist[mask]
        ports_block = port_seq[mask]
        lags_block  = lag_port_seq[mask]
        MI_2h[d], MI_24h[d] = mi_fn(H_block, ports_block, lags_block)

    return MI_2h, MI_24h


def ci_from_surrogates(surr: np.ndarray, upper: float = 99, lower: float = 1) -> Dict[str, np.ndarray]:
    surr = np.asarray(surr)
    low  = np.percentile(surr, lower, axis=0)
    high = np.percentile(surr, upper, axis=0)
    mean = np.nanmean(surr, axis=0)
    return {"low": low, "high": high, "mean": mean}


def plot_real_vs_sim_mi(
    MI_real_2h: np.ndarray, MI_real_24h: np.ndarray,
    MI_sim_2h: np.ndarray,  MI_sim_24h: np.ndarray,
    CI_real_2h: Optional[dict] = None,
    CI_real_24h: Optional[dict] = None,
    use_same_hue: bool = True,
    title: str = 'Real vs Sim Memory Index',
    ylim: Tuple[float, float] = (-0.25, 0.45)
) -> Tuple[plt.Figure, plt.Axes]:
    x = np.arange(len(MI_real_2h))
    fig, ax = plt.subplots(figsize=(6, 4))

    if use_same_hue:
        col_2h  = 'C0'
        col_24h = 'C1'
        sim_col_2h, sim_col_24h = col_2h, col_24h
    else:
        col_2h, col_24h = 'C0', 'C1'
        sim_col_2h, sim_col_24h = '0.25', '0.55'

    if CI_real_2h is not None:
        ax.fill_between(x, CI_real_2h['low'], CI_real_2h['high'], alpha=0.10, lw=0, color=col_2h)
        ax.plot(x, CI_real_2h['mean'], linestyle=':', alpha=0.5, color=col_2h)
    if CI_real_24h is not None:
        ax.fill_between(x, CI_real_24h['low'], CI_real_24h['high'], alpha=0.10, lw=0, color=col_24h)
        ax.plot(x, CI_real_24h['mean'], linestyle=':', alpha=0.5, color=col_24h)

    ax.plot(x, MI_real_2h,  '-', marker='o', ms=6, lw=1.8, color=col_2h,  label='Real 2h')
    ax.plot(x, MI_real_24h, '-', marker='s', ms=6, lw=1.8, color=col_24h, label='Real 24h')
    ax.plot(x, MI_sim_2h,  '--', marker='o', ms=6, mfc='white', mec=sim_col_2h, lw=1.5,
            color=sim_col_2h, label='Sim 2h')
    ax.plot(x, MI_sim_24h, '--', marker='s', ms=6, mfc='white', mec=sim_col_24h, lw=1.5,
            color=sim_col_24h, label='Sim 24h')

    ax.axhline(0, color='0.7', lw=1)
    ax.set_xticks(x)
    ax.set_xlabel('Distance to Yesterday Port')
    ax.set_ylabel('Memory Index')
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False, ncols=2)
    fig.tight_layout()
    return fig, ax


# -------------------------------------------------------------
# Histogram comparison (Real vs Sim) — with section bands + KL
# -------------------------------------------------------------
def hist_data_distance(Hist_data, port_seq, yes_port_seq, sess_distance):
    """
    Align histograms by 'distance bin' using your port/yes-port rotation & flip
    convention, keeping original semantics.
    """
    block = np.asarray(Hist_data[sess_distance], dtype=float)
    out = np.full_like(block, np.nan, dtype=float)
    axis_ports = block.ndim - 2

    for ss in range(len(port_seq[sess_distance])):
        p = port_seq[sess_distance][ss]
        y = yes_port_seq[sess_distance][ss]
        if np.isnan(p) or np.isnan(y):
            continue

        p = int(p); y = int(y)
        rolled = np.roll(block[ss], 4 - p, axis=axis_ports)
        cond = (
            (p > y and (y > 4 and (p - y) <= 3)) or
            (p < y and (y - p) >= 5)
        )
        if cond:
            rolled = np.flip(rolled, axis=axis_ports)
            rolled = np.roll(rolled, -1, axis=axis_ports)
        out[ss] = rolled
    return out


def plot_histograms_minimum_compare(
    Hist_data1, Hist_data2,
    port_seq, lag_port_seq, distance_seq,
    color1='C0', color2='C3',
    ymax=0.5, sess_type="",
    label1='Real', label2='Simulated'
):
    """
    Real vs Sim histogram comparison by distance bin, with KL per bin.
    """
    Hist_data1 = np.asarray(Hist_data1)
    Hist_data2 = np.asarray(Hist_data2)

    def get_histograms(Hist_data):
        masks = [np.array(distance_seq) == d for d in range(5)]
        return [hist_data_distance(Hist_data, port_seq, lag_port_seq, m) for m in masks]

    h1 = get_histograms(Hist_data1)
    h2 = get_histograms(Hist_data2)

    fig, axs = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axs = axs.flatten()

    x = np.arange(8)
    width = 0.35
    kl_divergences = []

    def process(hist):
        hist = np.asarray(hist, dtype=float)
        if hist.ndim != 2 or hist.shape[1] != 8:
            hist = hist.reshape(-1, 8)
        totals = np.nansum(hist, axis=1)
        valid = totals > 0
        if np.sum(valid) == 0:
            return np.zeros(8), np.zeros((2, 8))
        norm = hist[valid] / totals[valid][:, None]
        mean = np.nanmean(norm, axis=0)
        se = _sem(norm, axis=0, nan_policy='omit')
        yerr = np.vstack([se, se])
        return mean, yerr

    for i in range(5):
        m1, e1 = process(h1[i])
        m2, e2 = process(h2[i])

        kl = _kl(m1 + 1e-12, m2 + 1e-12)
        kl_divergences.append(kl)

        ax = axs[i]
        for left in (-0.5, 3.5):
            ax.axvspan(left, left + 2.0, alpha=0.06, color='0.2', zorder=0)

        ax.bar(x - width/2, m1, width=width, color=color1, alpha=0.9,
               label=label1 if i == 0 else None,
               yerr=e1, capsize=4, ecolor='k', error_kw=dict(lw=1), zorder=2)
        ax.bar(x + width/2, m2, width=width, color=color2, alpha=0.9,
               label=label2 if i == 0 else None,
               yerr=e2, capsize=4, ecolor='k', error_kw=dict(lw=1), zorder=2)

        ax.set_ylim(0, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels([-3, -2, -1, 0, 1, 2, 3, 4])
        ax.set_title("Real vs Sim — Histogram Alignment\n"
                     f"Distance {i}   |   KL={kl:.3f}", fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        if i % 3 == 0:
            ax.set_ylabel('Density')
        ax.set_xlabel('Relative position')

    axs[-1].axis('off')
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc='center', frameon=False)

    fig.suptitle(f'Histograms Comparison — {sess_type}', fontsize=16, fontweight='bold')
    return fig, kl_divergences


# =============================================================
#  NEW: compute_memory_metrics (no plotting, no PDFs)
# =============================================================
def compute_memory_metrics(
    # Transition / MI inputs
    real_sequences: Sequence[Sequence[int]],
    sim_sequences: Sequence[Sequence[int]],
    real_firstpoke_seqs: Sequence[Sequence[int]],
    sim_firstpoke_seqs: Sequence[Sequence[int]],
    real_sessions_trials: Sequence[Sequence[Sequence[int]]],
    sim_sessions_trials: Sequence[Sequence[Sequence[int]]],
    BH_real: np.ndarray,
    BH_sim: np.ndarray,
    ports: np.ndarray,
    lags: np.ndarray,
    dists: np.ndarray,
    mi_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]],
    surrogate_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    surrogate_mode: str = 'Shuffle',
    surrogate_n: int = 100,
) -> Dict[str, Any]:
    """
    Compute all comparison metrics WITHOUT generating figures or PDFs.
    This is for scripts / LL-based comparisons etc.
    """
    # Transition matrices
    n_ports = 8
    C_whole_real = transition_counts(real_sequences, n_ports)
    C_whole_sim  = transition_counts(sim_sequences,  n_ports)
    P_whole_real = row_normalize_probs(C_whole_real)
    P_whole_sim  = row_normalize_probs(C_whole_sim)

    C_fp_real = firstpoke_transition_counts(real_firstpoke_seqs, n_ports)
    C_fp_sim  = firstpoke_transition_counts(sim_firstpoke_seqs,  n_ports)
    P_fp_real = row_normalize_probs(C_fp_real)
    P_fp_sim  = row_normalize_probs(C_fp_sim)

    C_lf_real = lastfirst_transition_counts(real_sessions_trials, n_ports)
    C_lf_sim  = lastfirst_transition_counts(sim_sessions_trials,  n_ports)
    P_lf_real = row_normalize_probs(C_lf_real)
    P_lf_sim  = row_normalize_probs(C_lf_sim)

    # Histograms KL by distance
    _, KLs = plot_histograms_minimum_compare(BH_real, BH_sim, ports, lags, dists)
    plt.close()  # don't show when used in pure-metric mode

    # MI real vs sim + optional CIs
    MI_real_2h,  MI_real_24h  = compute_mi_by_distance(BH_real, ports, lags, dists, mi_fn=mi_fn)
    MI_sim_2h,   MI_sim_24h   = compute_mi_by_distance(BH_sim,  ports, lags, dists, mi_fn=mi_fn)

    CI_2h = None
    CI_24h = None
    MI_Surr2h_dist = None
    MI_Surr24h_dist = None
    if surrogate_fn is not None:
        MI_Surr2h_dist, MI_Surr24h_dist = surrogate_fn(
            BH_real, ports, lags, dists, surrogate_mode, surrogate_n
        )
        CI_2h  = ci_from_surrogates(MI_Surr2h_dist, upper=99, lower=1)
        CI_24h = ci_from_surrogates(MI_Surr24h_dist, upper=99, lower=1)

    return {
        "P_whole_real": P_whole_real,
        "P_whole_sim":  P_whole_sim,
        "P_fp_real":    P_fp_real,
        "P_fp_sim":     P_fp_sim,
        "P_lf_real":    P_lf_real,
        "P_lf_sim":     P_lf_sim,
        "MI_real_2h":   MI_real_2h,
        "MI_real_24h":  MI_real_24h,
        "MI_sim_2h":    MI_sim_2h,
        "MI_sim_24h":   MI_sim_24h,
        "MI_Surr2h_dist": MI_Surr2h_dist,
        "MI_Surr24h_dist": MI_Surr24h_dist,
        "CI_2h":        CI_2h,
        "CI_24h":        CI_24h,
        "KLs":          np.array(KLs),
    }


# =============================================================
#  PDF report (built on top of compute_memory_metrics)
# =============================================================
def generate_memory_report(
    # F-theta params
    today_port: int,
    yesterday_port: int,
    k_today: float,
    k_yesterday: float,
    w: float,
    # Transition inputs
    real_sequences: Sequence[Sequence[int]],
    sim_sequences: Sequence[Sequence[int]],
    real_firstpoke_seqs: Sequence[Sequence[int]],
    sim_firstpoke_seqs: Sequence[Sequence[int]],
    real_sessions_trials: Sequence[Sequence[Sequence[int]]],
    sim_sessions_trials: Sequence[Sequence[Sequence[int]]],
    # MI inputs
    BH_real: np.ndarray,
    BH_sim: np.ndarray,
    ports: np.ndarray,
    lags: np.ndarray,
    dists: np.ndarray,
    mi_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], Tuple[float, float]],
    surrogate_fn: Optional[Callable[..., Tuple[np.ndarray, np.ndarray]]] = None,
    surrogate_mode: str = 'Shuffle',
    surrogate_n: int = 100,
    # Output
    output_pdf_path: Optional[str] = "memory_report.pdf",
    n_ports: int = 8,
    title_prefix: str = "Memory Report",
    # Optional LL metrics for appendix
    ll_metrics: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Full PDF report (if output_pdf_path is not None) + returns metrics dict.

    To only get metrics (no figures, no PDF), use compute_memory_metrics(...)
    instead.
    """
    # 0) Compute all metrics first
    metrics = compute_memory_metrics(
        real_sequences=real_sequences,
        sim_sequences=sim_sequences,
        real_firstpoke_seqs=real_firstpoke_seqs,
        sim_firstpoke_seqs=sim_firstpoke_seqs,
        real_sessions_trials=real_sessions_trials,
        sim_sessions_trials=sim_sessions_trials,
        BH_real=BH_real,
        BH_sim=BH_sim,
        ports=ports,
        lags=lags,
        dists=dists,
        mi_fn=mi_fn,
        surrogate_fn=surrogate_fn,
        surrogate_mode=surrogate_mode,
        surrogate_n=surrogate_n,
    )

    MI_real_2h  = metrics["MI_real_2h"]
    MI_real_24h = metrics["MI_real_24h"]
    MI_sim_2h   = metrics["MI_sim_2h"]
    MI_sim_24h  = metrics["MI_sim_24h"]
    CI_2h       = metrics["CI_2h"]
    CI_24h      = metrics["CI_24h"]
    KLs         = metrics["KLs"]

    # 1) F_theta fig
    ports_1b, f_theta = compute_f_theta(
        n_ports=n_ports,
        today_port=today_port,
        yesterday_port=yesterday_port,
        k_today=k_today,
        k_yesterday=k_yesterday,
        w=w,
    )
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    plot_f_theta(ax1, ports_1b, f_theta, today_port, yesterday_port, title="F_theta")
    fig1.tight_layout()

    # 2) Transition matrices fig
    P_whole_real = metrics["P_whole_real"]
    P_whole_sim  = metrics["P_whole_sim"]
    P_fp_real    = metrics["P_fp_real"]
    P_fp_sim     = metrics["P_fp_sim"]
    P_lf_real    = metrics["P_lf_real"]
    P_lf_sim     = metrics["P_lf_sim"]

    vmax_global = max(P_whole_real.max(), P_whole_sim.max(),
                      P_fp_real.max(),    P_fp_sim.max(),
                      P_lf_real.max(),    P_lf_sim.max())
    vmin_global = 0.0

    fig2, axes = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    plot_prob(axes[0, 0], P_whole_real, "Within-trial — REAL", vmin_global, vmax_global)
    plot_prob(axes[0, 1], P_whole_sim,  "Within-trial — SIM",  vmin_global, vmax_global)
    plot_prob(axes[1, 0], P_fp_real,    "First-poke (trial→trial) — REAL", vmin_global, vmax_global)
    plot_prob(axes[1, 1], P_fp_sim,     "First-poke (trial→trial) — SIM",  vmin_global, vmax_global)
    plot_prob(axes[2, 0], P_lf_real,    "Last→First (t→t+1) — REAL", vmin_global, vmax_global)
    plot_prob(axes[2, 1], P_lf_sim,     "Last→First (t→t+1) — SIM",  vmin_global, vmax_global)
    fig2.colorbar(axes[0, 0].images[0], ax=axes, orientation="vertical",
                  fraction=0.03, pad=0.04, label="Probability")
    fig2.suptitle("Transition probability matrices (shared scale)", fontsize=13)

    # 3) Histograms comparison fig
    fig3, KLs_for_fig = plot_histograms_minimum_compare(
        BH_real, BH_sim,
        ports, lags, dists,
        color1='C0', color2='C3',
        ymax=0.5, sess_type="Recall", label1="Real", label2="Sim"
    )

    # 4) MI fig
    fig4, ax4 = plot_real_vs_sim_mi(
        MI_real_2h, MI_real_24h, MI_sim_2h, MI_sim_24h,
        CI_real_2h=CI_2h, CI_real_24h=CI_24h,
        use_same_hue=True, title="Real vs Sim Memory Index"
    )

    # --- Write PDF only if requested
    if output_pdf_path is not None:
        os.makedirs(os.path.dirname(output_pdf_path) or ".", exist_ok=True)
        with PdfPages(output_pdf_path) as pdf:
            # Cover / title page
            fig0 = plt.figure(figsize=(8.5, 11))
            fig0.text(0.5, 0.85, title_prefix, ha="center", va="center", fontsize=20, weight="bold")
            meta = [
                f"Today port: {today_port}   Yesterday port: {yesterday_port}",
                f"k_today={k_today:.3g}   k_yesterday={k_yesterday:.3g}   w={w:.3g}",
                f"N(real sessions)={BH_real.shape[0]}   N(sim sessions)={BH_sim.shape[0]}",
            ]
            for i, line in enumerate(meta):
                fig0.text(0.5, 0.75 - 0.03*i, line, ha="center", va="center", fontsize=12)
            pdf.attach_note("Generated by memory_report.py")
            pdf.savefig(fig0); plt.close(fig0)

            pdf.savefig(fig1); plt.close(fig1)
            pdf.savefig(fig2); plt.close(fig2)
            pdf.savefig(fig3); plt.close(fig3)
            pdf.savefig(fig4); plt.close(fig4)

            # 5) Appendix — metrics summary ( + optional LL )
            fig5 = plt.figure(figsize=(8.5, 11))
            plt.axis('off')

            def _fmt(arr: np.ndarray) -> str:
                return ", ".join([f"{v:.3f}" if np.isfinite(v) else "nan" for v in arr])

            mi_lines = [
                "Memory Index by distance (0..4):",
                f"  Real  2h: [{_fmt(MI_real_2h)}]   mean={np.nanmean(MI_real_2h):.3f}",
                f"  Real 24h: [{_fmt(MI_real_24h)}]   mean={np.nanmean(MI_real_24h):.3f}",
                f"  Sim   2h: [{_fmt(MI_sim_2h)}]    mean={np.nanmean(MI_sim_2h):.3f}",
                f"  Sim  24h: [{_fmt(MI_sim_24h)}]   mean={np.nanmean(MI_sim_24h):.3f}",
                "",
                "KL divergence per distance (Real ‖ Sim):",
                "  " + ", ".join([f"{v:.3f}" for v in KLs]) + f"   (mean={np.nanmean(KLs):.3f})",
            ]

            if ll_metrics is not None:
                mi_lines.append("")
                mi_lines.append("Log-likelihood metrics:")
                for name, arr in ll_metrics.items():
                    mi_lines.append(
                        f"  {name}: mean={np.nanmean(arr):.3f}, median={np.nanmedian(arr):.3f}"
                    )

            y = 0.9
            fig5.text(0.5, y, "Appendix — Metrics Summary", ha="center", fontsize=18, weight="bold")
            y -= 0.05
            for line in mi_lines:
                fig5.text(0.1, y, line, ha="left", va="top", fontsize=12)
                y -= 0.035

            pdf.savefig(fig5); plt.close(fig5)

    # Return metrics (so you can compare across models/simulations)
    metrics["output_pdf_path"] = output_pdf_path
    return metrics


# =============================================================
#  Within-trial split-by-length plot (you already had this)
# =============================================================
def count_trials_by_length(seqs, L_max):
    ctr = Counter(len(tr) for sess in seqs for tr in sess)
    return {L: ctr.get(L, 0) for L in range(1, L_max+1)}


def plot_within_trial_bars_real_vs_sim_counts_column(
    stats_real: Dict[int, Dict[str, np.ndarray]],
    stats_sim: Dict[int, Dict[str, np.ndarray]],
    real_sequences,
    sim_sequences,
    L_max: int = 4,
    K_max: int = 4,
    suptitle: str = "Within-trial (split by length): REAL vs SIM — fixed x-axis"
):
    """
    stats_real / stats_sim:
      output of your _within_trial_by_length_pooled(seqs, correct, yesterday, L_max, K_max)
      they must contain, per L:
        stats[L]["props"] -> (K_max, 3) probabilities for [today, yesterday, rest]
    """
    color_today = "#F39C12"   # orange
    color_yest  = "#1F77B4"   # blue
    color_rest  = "#2ECC71"   # green
    cat_colors  = [color_today, color_yest, color_rest]
    cat_labels  = ["today’s", "yesterday’s", "rest of ports"]

    n_real = count_trials_by_length(real_sequences, L_max)
    n_sim  = count_trials_by_length(sim_sequences,  L_max)
    max_n  = max(1, max([n_real.get(L,0) for L in range(1,L_max+1)] +
                         [n_sim.get(L,0)  for L in range(1,L_max+1)]))

    n_rows, n_cols = L_max, K_max + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2*n_cols, 3.7*n_rows), squeeze=False)

    cat_x = np.arange(3)
    width = 0.35

    for L in range(1, L_max+1):
        for k in range(1, K_max+1):
            ax = axes[L-1, k-1]
            if k > L:
                ax.axis("off")
                continue

            P_real = stats_real.get(L, {}).get("props", np.full((min(K_max,L),3), np.nan))
            P_sim  = stats_sim.get(L,  {}).get("props", np.full((min(K_max,L),3), np.nan))
            pr = P_real[k-1] if k-1 < P_real.shape[0] else np.array([np.nan]*3)
            ps = P_sim[k-1]  if k-1 < P_sim.shape[0]  else np.array([np.nan]*3)

            x_real = cat_x - width/2
            x_sim  = cat_x + width/2

            for i, (xr, xs) in enumerate(zip(x_real, x_sim)):
                ax.bar(xr, pr[i], width=width, color=cat_colors[i], alpha=0.95, edgecolor="black", linewidth=0.6)
                ax.bar(xs, ps[i], width=width, color=cat_colors[i], alpha=0.40, edgecolor="black", linewidth=0.6)

            ax.axhline(0.125, color="grey", linestyle=":", linewidth=1.1)
            ax.axhline(0.750, color="grey", linestyle=":", linewidth=1.1)

            ax.set_xlim(-0.7, 2.7)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(cat_x)
            ax.set_xticklabels(cat_labels, rotation=20)
            if k == 1:
                ax.set_ylabel(f"{L} poke{'s' if L>1 else ''}/trial\nfraction of trials")
            if L == 1:
                ord_map = {1:"1st",2:"2nd",3:"3rd"}
                ax.set_title(f"{ord_map.get(k,str(k)+'th')} poke")

        # rightmost column: counts
        axc = axes[L-1, K_max]
        axc.barh(["SIM","REAL"],
                 [n_sim.get(L,0), n_real.get(L,0)],
                 color=["#C9C9C9","#111111"], height=0.55)
        axc.set_xlim(0, max_n+100)
        axc.set_xlabel("trials")
        axc.set_yticks([0,1]); axc.set_yticklabels(["SIM","REAL"])
        axc.grid(axis="x", alpha=0.2)
        for y, v in enumerate([n_sim.get(L,0), n_real.get(L,0)]):
            axc.text(v + max_n*0.02, y, str(v), va="center", ha="left", fontsize=10)
        if L == 1:
            axc.set_title("trial counts", fontsize=11)

    cat_patches = [Patch(facecolor=c, edgecolor="black", label=lab, alpha=0.95)
                   for c, lab in zip(cat_colors, cat_labels)]
    dataset_patches = [Patch(facecolor="black", label="REAL (count)", alpha=0.95),
                       Patch(facecolor="#C9C9C9", label="SIM (count)", alpha=1.0)]
    fig.legend(cat_patches + dataset_patches,
               cat_labels + ["REAL (count)","SIM (count)"],
               ncol=5, loc="upper center", frameon=False, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(suptitle, y=1.10, fontsize=13)
    fig.tight_layout()
    plt.show()
