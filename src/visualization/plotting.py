import matplotlib.pyplot as plt

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
        rotated_hist_seq = np.roll(hist_data[i], 3 - int(arr_ports[i]))
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