import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_strategy_evolution(strategy_counts):
    """
    Plot the mean evolution of the three strategies over trials.

    Parameters:
        strategy_counts (np.ndarray): Array of shape (n_trials, 3) containing strategy probabilities.
    """
    plt.figure(figsize=(10, 6))

    # Plot the strategy lines
    plt.plot(range(strategy_counts.shape[0]), strategy_counts[:, 0], label="Exploration", linestyle='dashed', color='green')
    plt.plot(range(strategy_counts.shape[0]), strategy_counts[:, 1], label="Correct Port", linestyle='solid', color='blue')
    plt.plot(range(strategy_counts.shape[0]), strategy_counts[:, 2], label="Yesterday's Correct Port", linestyle='dotted', color='red')

    # Fill the intervals with the color of the line that has the max value during trials
    for i in range(strategy_counts.shape[0]):
        max_index = strategy_counts[i].argmax()
        if max_index == 0:
            plt.axvspan(i, i + 1, facecolor='green', alpha=0.1)
        elif max_index == 1:
            plt.axvspan(i, i + 1, facecolor='blue', alpha=0.1)
        elif max_index == 2:
            plt.axvspan(i, i + 1, facecolor='red', alpha=0.1)

    plt.xlabel("Trials")
    plt.ylabel("Mean Strategy Probability")
    plt.xlim(1, strategy_counts.shape[0] - 1)
    plt.title("Strategy Evolution Across Trials")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_histogram(hist_data, arr_ports):
    """
    Plot a histogram of poked ports centered at the correct port.

    Parameters:
        hist_data (list of np.ndarray): List of histogram data for each session.
        arr_ports (list): List of correct ports for each session.

    Returns:
        np.ndarray: Normalized histogram data.
    """
    rotated = []
    for i in range(len(hist_data)):
        rotated_hist_seq = np.roll(hist_data[i], 4 - int(arr_ports[i]))
        rotated.append(rotated_hist_seq)

    hist = np.sum(rotated, axis=0)
    normalized_hist = hist / np.sum(hist)
    bins = np.arange(-3, 5)  # Create bins from -3 to 4

    # Plot histogram using bar plot
    plt.bar(bins, normalized_hist, edgecolor='black', alpha=0.7)
    plt.xlabel('Ports')
    plt.ylabel('Frequency')
    plt.title('Histogram of Frequencies')
    plt.xticks(bins)
    plt.show()

    return normalized_hist

def plot_comparison_histograms(real_hist, simulated_hists, labels, bins=np.arange(-3, 5)):
    """
    Plot comparison of real and simulated histograms with consistent color assignment.

    Parameters:
        real_hist (np.ndarray): Normalized histogram of real data.
        simulated_hists (list of np.ndarray): List of normalized histograms for simulated data.
        labels (list of str): List of labels for the histograms (real and simulated).
        bins (np.ndarray): Bin edges for the histograms.
    """
    # Define consistent colors for models
    colors = {
        "Real Distribution": "red",
        "Simulated RL Distribution": "blue",
        "Simulated RL Distribution (belief = 2 deltas)": "green",
        "Simulated Normal Distribution Marginal Belief": "purple",
        "Simulated Normal Distribution Terminal Probability": "orange"
    }

    plt.figure(figsize=(10, 6))

    # Plot real histogram
    plt.plot(bins, real_hist, marker='o', color=colors[labels[0]], label=labels[0], linewidth=2)
    plt.bar(bins, real_hist, width=0.4, label=f'{labels[0]} (Bar)', align='center', color=colors[labels[0]], alpha=0.3)

    # Plot simulated histograms
    for sim_hist, label in zip(simulated_hists, labels[1:]):
        plt.plot(bins, sim_hist, marker='o', color=colors[label], label=label, linewidth=2)

    # Customize plot
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.xlabel('Ports')
    plt.ylabel('Frequency')
    plt.title('Comparison of Normalized and Simulated Histograms')
    plt.legend()
    plt.show()




def plot_distances_histograms(real_distances, simulated_distances, labels, bins=np.arange(-3, 5)):
    """
    Plot histograms for each distance category, comparing real and simulated data.

    Parameters:
        real_distances (dict): Dictionary where keys are distance categories (e.g., 0-4) and values are lists of real histogram data.
        simulated_distances (dict): Dictionary where keys are distance categories (e.g., 0-4) and values are lists of simulated histogram data.
        labels (list of str): List of labels for the histograms (real and simulated).
        bins (np.ndarray): Bin edges for the histograms.
    """
    # Define consistent colors for models
    colors = {
        "Real Distribution": "red",
        "Simulated RL Distribution": "blue",
        "Simulated RL Distribution (belief = 2 deltas)": "green",
        "Simulated Normal Distribution Marginal Belief": "purple",
        "Simulated Normal Distribution Terminal Probability": "orange"
    }

    fig, axs = plt.subplots(len(real_distances), 1, figsize=(10, 20))

    for dist, ax in enumerate(axs):
        # Compute normalized histograms for real and simulated data
        real_hist_values = np.sum(real_distances[dist], axis=0) / np.sum(real_distances[dist])
        simulated_hists_values = [
            np.sum(simulated_distances[model][dist], axis=0) / np.sum(simulated_distances[model][dist])
            for model in labels[1:]
        ]

        # Plot real histogram
        ax.plot(bins, real_hist_values, marker='o', color=colors[labels[0]], label=labels[0], linewidth=2)
        ax.bar(bins, real_hist_values, width=0.4, label=f'{labels[0]} (Bar)', align='center', color=colors[labels[0]], alpha=0.3)

        # Plot simulated histograms
        for sim_hist_values, label in zip(simulated_hists_values, labels[1:]):
            ax.plot(bins, sim_hist_values, marker='o', color=colors[label], label=label, linewidth=2)

        # Customize plot
        ax.set_title(f'Histogram for Distance {dist}')
        ax.set_xlabel('Ports')
        ax.set_ylabel('Frequency')
        ax.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            borderaxespad=0,
            frameon=True
        )
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()




def plot_transition_matrices(real_matrix, simulated_matrices, labels, num_ports=8):
    """
    Plot heatmaps for real and simulated transition matrices.

    Parameters:
        real_matrix (np.ndarray): Transition matrix for real data.
        simulated_matrices (list of np.ndarray): List of transition matrices for simulated data.
        labels (list of str): List of labels for the matrices (real and simulated).
        num_ports (int): Number of ports (default is 8).
    """
    plt.figure(figsize=(20, 20))

    # Plot real transition matrix
    plt.subplot(3, 3, 5)
    sns.heatmap(real_matrix, annot=True, cmap="Blues", xticklabels=range(1, num_ports + 1), yticklabels=range(1, num_ports + 1))
    plt.title(labels[0])
    plt.xlabel("To Port")
    plt.ylabel("From Port")

    # Plot simulated transition matrices
    for i, (sim_matrix, label) in enumerate(zip(simulated_matrices, labels[1:])):
        plt.subplot(3, 3, i + 1)
        cmap = sns.color_palette("coolwarm", as_cmap=True)
        sns.heatmap(sim_matrix, annot=True, cmap=cmap, xticklabels=range(1, num_ports + 1), yticklabels=range(1, num_ports + 1))
        plt.title(label)
        plt.xlabel("To Port")
        plt.ylabel("From Port")

    plt.tight_layout()
    plt.show()

