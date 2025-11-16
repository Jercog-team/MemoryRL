# utils/viz.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, sem


def plot_prob(ax, M, title, cmap, vmin=None, vmax=None):
    """
    Plot a heatmap of a probability / count matrix with numbers inside.
    """
    M = np.asarray(M, dtype=float)

    im = ax.imshow(
        M, origin="upper", aspect="equal", interpolation="nearest",
        vmin=vmin, vmax=vmax, cmap=cmap
    )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("To port")
    ax.set_ylabel("From port")

    ax.set_xticks(range(M.shape[1]))
    ax.set_yticks(range(M.shape[0]))
    ax.set_xticklabels(range(1, M.shape[1] + 1))
    ax.set_yticklabels(range(1, M.shape[0] + 1))

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}",
                    ha="center", va="center", fontsize=8)

    return im


def rotate_histograms_by_distance(Hist_data, port_seq, yes_port_seq, sess_distance):
    """
    Align (rotate/flip) histograms by distance and side relative to yesterday's port.

    Parameters
    ----------
    Hist_data : ndarray, shape (..., 8)
        Histogram data with a ports axis (length 8) at position Hist_data.ndim-2.
    port_seq : ndarray
        Current ports for each session/trial.
    yes_port_seq : ndarray
        'Yesterday' ports for each session/trial.
    sess_distance : ndarray (bool)
        Boolean mask selecting which sessions belong to the current distance bin.

    Returns
    -------
    Hist_data_dist : ndarray
        Same shape as Hist_data[sess_distance], but rotated (and sometimes flipped)
        so that the “distance to yesterday port” is aligned across sessions.
    """
    Hist_data_dist = np.full_like(Hist_data[sess_distance], np.nan)

    for ss in range(len(port_seq[sess_distance])):
        p = port_seq[sess_distance][ss]
        yp = yes_port_seq[sess_distance][ss]

        # Condition to flip (left/right symmetry around yesterday port)
        if (
            (p > yp and (yp > 4 and (p - yp) <= 3))
            or (p < yp and (yp - p) >= 5)
        ):
            # Reverse: roll to center on p, flip, then roll one step
            Hist_data_dist[ss] = np.roll(
                np.flip(
                    np.roll(
                        Hist_data[sess_distance][ss],
                        4 - int(p),
                        axis=Hist_data.ndim - 2,
                    ),
                    axis=Hist_data.ndim - 2,
                ),
                -1,
                axis=Hist_data.ndim - 2,
            )
        else:
            # Simple rotation to center on current port
            Hist_data_dist[ss] = np.roll(
                Hist_data[sess_distance][ss],
                4 - int(p),
                axis=Hist_data.ndim - 2,
            )

    return Hist_data_dist

# Backwards-compat alias (so old code still works)
hist_data_distance = rotate_histograms_by_distance



def plot_histograms_real_vs_sim_by_distance(
    Hist_data_real, Hist_data_sim,
    port_seq, lag_port_seq, distance_seq,
    color_real, color_sim,
    ymax, sess_type
):
    """
    Compare within-trial histograms for REAL vs SIM across distance bins 0..4.

    Parameters
    ----------
    Hist_data_real, Hist_data_sim : array-like
        Histogram data for real and simulated sessions.
    port_seq : array-like
        Current ports (per session/trial).
    lag_port_seq : array-like
        Yesterday ports.
    distance_seq : array-like
        Distance between current and yesterday ports (0..4).
    color_real, color_sim : colors
        Colors for real/sim bars.
    ymax : float
        Y-axis max for barplots.
    sess_type : str
        Text label for figure title.

    Returns
    -------
    kl_divergences : list[float]
        KL(real || sim) per distance bin.
    """
    Hist_data_real = np.asarray(Hist_data_real)
    Hist_data_sim = np.asarray(Hist_data_sim)

    def get_histograms(Hist_data):
        sess_masks = [np.array(distance_seq) == d for d in range(5)]
        return [
            rotate_histograms_by_distance(Hist_data, port_seq, lag_port_seq, mask)
            for mask in sess_masks
        ]

    histograms_real = get_histograms(Hist_data_real)
    histograms_sim = get_histograms(Hist_data_sim)

    fig, axs = plt.subplots(2, 3, figsize=np.array([6.4 * 3, 4.8 * 2]) * 0.7)
    axs = axs.flatten()

    x = np.linspace(-np.pi, np.pi, num=8, endpoint=False) + np.pi / 4
    kl_divergences = []

    for i in range(5):
        def process(hist):
            session_totals = np.sum(hist, axis=1)
            valid = session_totals > 0
            if np.sum(valid) == 0:
                return np.zeros(8), np.zeros((2, 8))
            norm_all = hist[valid] / session_totals[valid][:, None]
            mean = np.nanmean(norm_all, axis=0)
            std_err_val = sem(norm_all, axis=0, nan_policy="omit")
            yerr = np.vstack([std_err_val, std_err_val])
            return mean, yerr

        norm_real, yerr_real = process(histograms_real[i])
        norm_sim, yerr_sim = process(histograms_sim[i])

        eps = 1e-10
        kl = entropy(norm_real + eps, norm_sim + eps)
        kl_divergences.append(kl)

        axs[i].bar(
            x - 0.1, norm_real, width=0.18, color=color_real, alpha=0.7, label="Real",
            yerr=yerr_real, capsize=4, ecolor="k", error_kw=dict(lw=1),
        )
        axs[i].bar(
            x + 0.1, norm_sim, width=0.18, color=color_sim, alpha=0.7, label="Simulated",
            yerr=yerr_sim, capsize=4, ecolor="k", error_kw=dict(lw=1),
        )

        axs[i].set_ylim(0, ymax)
        axs[i].set_title(f"Distance {i}\nKL={kl:.3f}", fontweight="bold")
        axs[i].spines[["top", "right"]].set_visible(False)
        axs[i].set_xlabel("Ports")
        if i % 3 == 0:
            axs[i].set_ylabel("Density counts")
        if i == 0:
            axs[i].legend()

    axs[-1].axis("off")

    fig.suptitle(
        f"Histograms Comparison REAL vs SIM\n{sess_type} Highest Log-Likelihood",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return kl_divergences

# Backwards-compat alias
plot_histograms_minimum_compare = plot_histograms_real_vs_sim_by_distance



def plot_histograms_3datasets_by_distance(
    Hist_data1, Hist_data2, Hist_data3,
    port_seq, lag_port_seq, distance_seq,
    color1, color2, color3,
    ymax=0.5, sess_type="",
    label1="A", label2="B", label3="C",
):
    """
    Compare histogram profiles for 3 datasets across distances 0..4.

    Returns dict with pairwise KL lists per distance: KL12, KL13, KL23.
    """
    Hist_data1 = np.asarray(Hist_data1)
    Hist_data2 = np.asarray(Hist_data2)
    Hist_data3 = np.asarray(Hist_data3)

    def get_histograms(Hist_data):
        sess_masks = [np.array(distance_seq) == d for d in range(5)]
        return [
            rotate_histograms_by_distance(Hist_data, port_seq, lag_port_seq, mask)
            for mask in sess_masks
        ]

    histograms1 = get_histograms(Hist_data1)
    histograms2 = get_histograms(Hist_data2)
    histograms3 = get_histograms(Hist_data3)

    fig, axs = plt.subplots(2, 3, figsize=np.array([6.4 * 3, 4.8 * 2]) * 0.7)
    axs = axs.flatten()

    x = np.linspace(-np.pi, np.pi, num=8, endpoint=False) + np.pi / 4
    KL12, KL13, KL23 = [], [], []

    for i in range(5):
        def process(hist):
            session_totals = np.sum(hist, axis=1)
            valid = session_totals > 0
            if np.sum(valid) == 0:
                return np.zeros(8), np.zeros((2, 8))
            norm_all = hist[valid] / session_totals[valid][:, None]
            mean = np.nanmean(norm_all, axis=0)
            std_err_val = sem(norm_all, axis=0, nan_policy="omit")
            yerr = np.vstack([std_err_val, std_err_val])
            return mean, yerr

        n1, e1 = process(histograms1[i])
        n2, e2 = process(histograms2[i])
        n3, e3 = process(histograms3[i])

        eps = 1e-10
        KL12.append(entropy(n1 + eps, n2 + eps))
        KL13.append(entropy(n1 + eps, n3 + eps))
        KL23.append(entropy(n2 + eps, n3 + eps))

        axs[i].bar(
            x - 0.20, n1, width=0.18, color=color1, alpha=0.7, label=label1,
            yerr=e1, capsize=4, ecolor="k", error_kw=dict(lw=1),
        )
        axs[i].bar(
            x + 0.00, n2, width=0.18, color=color2, alpha=0.7, label=label2,
            yerr=e2, capsize=4, ecolor="k", error_kw=dict(lw=1),
        )
        axs[i].bar(
            x + 0.20, n3, width=0.18, color=color3, alpha=0.7, label=label3,
            yerr=e3, capsize=4, ecolor="k", error_kw=dict(lw=1),
        )

        axs[i].set_ylim(0, ymax)
        axs[i].set_title(
            f"Distance {i}\n"
            f"KL {label1}|{label2}={KL12[-1]:.3f}, "
            f"{label1}|{label3}={KL13[-1]:.3f}, "
            f"{label2}|{label3}={KL23[-1]:.3f}",
            fontweight="bold",
        )
        axs[i].spines[["top", "right"]].set_visible(False)
        axs[i].set_xlabel("Ports")
        if i % 3 == 0:
            axs[i].set_ylabel("Density")

    axs[-1].axis("off")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[-1].legend(handles, labels, loc="center", frameon=False)

    fig.suptitle(
        f"Histograms Comparison (3 datasets)\n{sess_type}",
        fontsize=16, fontweight="bold",
    )
    plt.tight_layout()
    plt.show()

    return {"KL12": KL12, "KL13": KL13, "KL23": KL23}

# Backwards-compat alias
plot_histograms_minimum_compare3 = plot_histograms_3datasets_by_distance



def plot_session_trajectory_with_ports(
    normalized_traj,
    normalized_port_coordinates,
    positions_start_trial=None,
    sess: int = 0,
    ax=None,
    title: str | None = None,
):
    """
    Plot a single session's normalized trajectory on the unit circle, with ports
    and (optionally) the initial position of each trial.

    Parameters
    ----------
    normalized_traj : list of ndarray
        normalized_traj[s] has shape (T, 3): [x_norm, y_norm, t].
        These are the trajectories already normalized to the unit box/circle.
    normalized_port_coordinates : list of ndarray
        normalized_port_coordinates[s] has shape (N_PORTS, 2) with [x, y]
        coordinates of each port on the unit circle for session s.
    positions_start_trial : list of ndarray or None, optional
        positions_start_trial[s] has shape (n_cues, 2) with [x, y] positions
        at cue onset (trial start). If None or positions_start_trial[sess] is
        None/empty, the onset markers are skipped.
    sess : int, default 0
        Index of the session to visualize.
    ax : matplotlib.axes.Axes, optional
        Axis to draw into. If None, a new figure+axis is created.
    title : str, optional
        Title for the plot. If None, a default is used.

    Returns
    -------
    fig, ax : matplotlib.figure.Figure, matplotlib.axes.Axes
        The figure and axis objects used.
    """
    traj = np.asarray(normalized_traj[sess])
    ports = np.asarray(normalized_port_coordinates[sess])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    if title is None:
        title = f"Normalized Trajectory with Ports (session {sess})"

    # Draw unit circle
    circle = plt.Circle((0, 0), 1, color="r", fill=False, alpha=0.7, linewidth=1.0)
    ax.add_patch(circle)

    # Plot trajectory
    if traj.size > 0:
        ax.plot(traj[:, 0], traj[:, 1],
                label="Normalized Trajectory", alpha=0.6)

    # Plot ports (PoR)
    if ports.size > 0:
        ax.scatter(
            ports[:, 0], ports[:, 1],
            color="blue", s=80, label="Ports (PoR)", zorder=3,
        )

    # Plot initial trial positions if provided
    if positions_start_trial is not None:
        pos_sess = positions_start_trial[sess]
        if pos_sess is not None:
            pos_sess = np.asarray(pos_sess)
            if pos_sess.size > 0:
                ax.scatter(
                    pos_sess[:, 0], pos_sess[:, 1],
                    color="lime", s=40, label="Trial onsets",
                    zorder=4, edgecolor="k", linewidth=0.5,
                )

    # Cosmetics
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_data_based_MI(MI, MILags, MI_Surr2h_dist, MI_Surr24h_dist, colors):
    """
    Plot model-free (data-based) 2h and 24h memory index with surrogate CIs.
    """
    CI_MI2h_Surr = np.percentile(MI_Surr2h_dist, [99, 1], axis=0)
    CI_MI_SurrLags = np.percentile(MI_Surr24h_dist, [99, 1], axis=0)

    x_plot = np.arange(5)
    fig, axs = plt.subplots(1, 1, figsize=np.array([6.4, 6.4]) * 0.7)

    # 2h
    axs.scatter([x_plot[0]], [MI[0]], color=colors[0], alpha=0.4, s=7, zorder=3)
    axs.plot(x_plot[:2], MI[:2], "--", alpha=0.5, color=colors[0])
    axs.plot(x_plot[1:], MI[1:], ".-", alpha=0.5, color=colors[0], label="2h MI")

    # 24h
    axs.scatter([x_plot[0]], [MILags[0]], color=colors[1], alpha=0.4, s=7, zorder=3)
    axs.plot(x_plot[:2], MILags[:2], "--", alpha=0.5, color=colors[1])
    axs.plot(x_plot[1:], MILags[1:], ".-", alpha=0.5, color=colors[1], label="24h MI")

    # Surrogate bands
    axs.fill_between(x_plot, CI_MI2h_Surr[0], CI_MI2h_Surr[1], color=colors[0], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI2h_Surr, axis=0), "--", color=colors[0], alpha=0.1)
    axs.fill_between(x_plot, CI_MI_SurrLags[0], CI_MI_SurrLags[1], color=colors[1], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI_SurrLags, axis=0), "--", color=colors[1], alpha=0.1)

    axs.axhline(y=0, color="grey", alpha=0.2)
    axs.spines[["top", "right"]].set_visible(False)
    axs.set_ylabel("Memory Index")
    axs.set_xlabel("Distance to Yesterday Port")
    axs.set_title("Model-free Memory Index – Real Data")
    axs.set_ylim(-0.2, 0.4)
    axs.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=8)
    return fig, axs


# Backwards compat
figure_data_based_MI = plot_data_based_MI


def plot_model_based_MI(
    models, port_seq, lag_port_seq, distance_seq,
    MI_Surr2h_dist, MI_Surr24h_dist,
    colors, Transitions, SurrogateMode, DrugType,
    save=False, save_dir=r"D:\AutoSynaptopatiesData\Figures\HMM"
):
    """
    Plot model-based memory index (from HMM) + surrogate CIs.
    """
    from pathlib import Path

    MI_m, MILags_m = model_based_MI(models, port_seq, lag_port_seq, distance_seq)[:2]
    Kappas = [np.exp(models[distance]["model"].observations.log_kappas) for distance in range(5)]
    kappas = Kappas[0][1]

    CI_MI2h_Surr = np.percentile(MI_Surr2h_dist, [99, 1], axis=0)
    CI_MI_SurrLags = np.percentile(MI_Surr24h_dist, [99, 1], axis=0)

    x_plot = np.arange(5)
    fig, axs = plt.subplots(1, 1, figsize=np.array([6.4, 6.4]) * 0.7)

    # 2h
    axs.scatter([x_plot[0]], [MI_m[0]], color=colors[0], alpha=0.4, s=7, zorder=3)
    axs.plot(x_plot[:2], MI_m[:2], "--", alpha=0.5, color=colors[0])
    axs.plot(x_plot[1:], MI_m[1:], ".-", alpha=0.5, color=colors[0], label="Model Pred. 2h MI")

    # 24h
    axs.scatter([x_plot[0]], [MILags_m[0]], color=colors[1], alpha=0.4, s=7, zorder=3)
    axs.plot(x_plot[:2], MILags_m[:2], "--", alpha=0.5, color=colors[1])
    axs.plot(x_plot[1:], MILags_m[1:], ".-", alpha=0.5, color=colors[1], label="Model Pred. 24h MI")

    # Surrogate bands
    axs.fill_between(x_plot, CI_MI2h_Surr[0], CI_MI2h_Surr[1], color=colors[0], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI2h_Surr, axis=0), "--", color=colors[0], alpha=0.1)
    axs.fill_between(x_plot, CI_MI_SurrLags[0], CI_MI_SurrLags[1], color=colors[1], alpha=0.05)
    axs.plot(x_plot, np.nanmean(CI_MI_SurrLags, axis=0), "--", color=colors[1], alpha=0.1)

    axs.axhline(y=0, color="grey", alpha=0.2)
    axs.spines[["top", "right"]].set_visible(False)
    axs.set_ylabel("Memory Index")
    axs.set_xlabel("Distance to Yesterday Port")
    axs.set_title(f"Model-based Memory Index ({Transitions})\nκ={np.round(kappas[0])}")
    axs.set_ylim(-0.2, 0.4)
    axs.legend(loc="upper right", bbox_to_anchor=(1.3, 1), fontsize=8)

    if save:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fname = save_dir / f"Model_Based_MI_{DrugType}_{SurrogateMode}_{Transitions}.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")

    return fig, axs


figure_model_based_MI = plot_model_based_MI  # alias
