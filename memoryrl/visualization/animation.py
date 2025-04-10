from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt


def animate_beliefs_qs(animation_data, use_weighted=True):
    """
    Creates an animation showing the evolution of belief distributions, penalty masks, Q-values, and final action preferences.

    Parameters:
        animation_data (list of dict):
            A list of dictionaries where each dictionary represents a step in the animation. Each dictionary should contain the following keys:
                - 'trial': The trial number.
                - 'poke': The poke number.
                - 'marginal_belief': The belief distribution at the current step.
                - 'q_values': The raw Q-values at the current step.
                - 'weighted_q': The weighted Q-values after applying penalties.
                - 'chosen_action': The index of the chosen action.
                - 'last_action': The index of the last action taken.
                - 'penalty_mask': The penalty mask applied to Q-values.
                - 'exploration' (optional): A boolean indicating whether the step is exploratory.
        use_weighted (bool, optional):
            If True, the animation will display weighted Q-values in the final action preference plot. Defaults to True.

    Returns:
        IPython.core.display.HTML: An HTML object containing the animation.

    Example:
        >>> from src.visualization.animation import animate_beliefs_qs
        >>> animation_data = [
        ...     {
        ...         'trial': 1,
        ...         'poke': 1,
        ...         'marginal_belief': [0.1, 0.2, 0.3, 0.4],
        ...         'q_values': [0.5, 0.6, 0.7, 0.8],
        ...         'weighted_q': [0.4, 0.5, 0.6, 0.7],
        ...         'chosen_action': 2,
        ...         'last_action': 1,
        ...         'penalty_mask': [-1, -2, -3, -4],
        ...         'exploration': True
        ...     }
        ... ]
        >>> html_animation = animate_beliefs_qs(animation_data)
        >>> display(html_animation)
    """
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), constrained_layout=True)
    n_ports = len(animation_data[0]['marginal_belief'])

    def update(i):
        for ax in axs:
            ax.clear()

        step = animation_data[i]
        trial = step["trial"]
        poke = step["poke"]
        m_belief = step["marginal_belief"]
        q_vals = step["q_values"]
        weighted_q = step["weighted_q"]
        chosen = step["chosen_action"]
        last_action = step["last_action"]
        is_exploration = step.get("exploration", False)

        # 1. Beliefs
        axs[0].bar(range(n_ports), m_belief, color="skyblue")
        axs[0].set_title("Belief Distribution: $m_t(r)$", fontsize=12)
        axs[0].set_ylim(0, 1)

        # 2. Actual penalty mask applied to Q-values
        penalty_mask = step["penalty_mask"]
        axs[1].bar(range(n_ports), penalty_mask, color="orange")
        axs[1].set_title("Penalty Mask Applied to Q-values", fontsize=12)
        axs[1].set_ylim(min(penalty_mask.min(), -6), max(penalty_mask.max(), 1))


        # 3. Raw Q-values
        axs[2].bar(range(n_ports), q_vals, color='gray')
        axs[2].set_title("Raw Q-values", fontsize=12)
        axs[2].set_ylim(min(q_vals) - 0.1, max(q_vals) + 0.1)

        # 4. Final preference
        # 4. Final preference with color based on choice type
        data_to_plot = weighted_q if use_weighted else q_vals
        colors = ['red' if is_exploration and i == chosen else
                'green' if i == chosen else
                'blue' for i in range(n_ports)]

        axs[3].bar(range(n_ports), data_to_plot, color=colors)
        axs[3].set_title("Final Action Preference (After Penalty × Q)", fontsize=12)
        axs[3].set_ylim(min(data_to_plot) - 0.1, max(data_to_plot) + 0.1)
        axs[3].set_xticks(range(n_ports))
        axs[3].set_ylabel("Value")

        # Label exploration status clearly
        axs[3].text(0.5, 0.9, 
                    "Exploration" if is_exploration else "Policy-guided",
                    transform=axs[3].transAxes,
                    ha='center', fontsize=11,
                    color='red' if is_exploration else 'green')


        # Common formatting
        for ax in axs:
            ax.set_xticks(range(n_ports))
            ax.set_ylabel("Value")

        axs[3].set_xlabel(f"Trial {trial}, Poke {poke} {'(Exploration)' if is_exploration else ''}", fontsize=11)

    ani = FuncAnimation(fig, update, frames=len(animation_data), interval=400, repeat=False)
    return HTML(ani.to_jshtml())
