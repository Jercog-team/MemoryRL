import numpy as np
import random
import matplotlib.pyplot as plt

class MouseBehaviorPOMDP:
    def __init__(self, num_ports=8, observation_noise=0.1, discount_factor=0.95, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        self.num_ports = num_ports
        self.states = list(range(num_ports))  # States represent ports
        self.actions = list(range(num_ports))  # Actions are choosing a port
        self.discount_factor = discount_factor
        self.observation_noise = observation_noise
        
        # Observations
        self.observations = {
            "distance": list(range(num_ports)),  # Distance to the correct port
            "reward_available": [0, 1],  # Whether water is available
        }
        
        # Initialize transition, observation, and reward matrices
        self.T = self._initialize_transitions()
        self.O = self._initialize_observations()
        self.R = self._initialize_rewards()
        
        # Dynamic variables
        self.reset()

    def _initialize_transitions(self):
        """Initialize state transition probabilities."""
        T = np.zeros((self.num_ports, self.num_ports, self.num_ports))
        for s in self.states:
            for a in self.actions:
                for s_next in self.states:
                    # Fixed state transition for simplicity (stationary target port)
                    T[s, a, s_next] = 1.0 if s_next == s else 0.0
        return T

    def _initialize_observations(self):
        """Initialize observation probabilities."""
        O = {
            "distance": np.zeros((self.num_ports, self.num_ports, self.num_ports)),
            "reward_available": np.zeros((self.num_ports, self.num_ports, 2)),
        }
        
        for s in self.states:
            for a in self.actions:
                # Distance observation
                for s_next in self.states:
                    O["distance"][s, a, s_next] = 1.0 - (abs(s - s_next) / self.num_ports)
                
                # Reward availability
                O["reward_available"][s, a, 1] = 0.9 if s == a else 0.1  # Reward likely if correct action
                O["reward_available"][s, a, 0] = 1.0 - O["reward_available"][s, a, 1]
        
        # Normalize distance probabilities
        for s in self.states:
            for a in self.actions:
                O["distance"][s, a, :] /= np.sum(O["distance"][s, a, :])
        
        return O

    def _initialize_rewards(self):
        """Initialize the reward structure."""
        R = np.zeros((self.num_ports, self.num_ports))
        for s in self.states:
            for a in self.actions:
                if s == a:
                    R[s, a] = 1.0  # Reward when water is available
                elif abs(s - a) == 1:
                    R[s, a] = 0.5  # Slight reward for adjacent ports
                elif abs(s - a) == 2:
                    R[s, a] = 0.3  # Even smaller reward for two-away ports
                else:
                    R[s, a] = 0.0  # No reward for other ports
        return R

    def reset(self, keep_belief=False):
        """Reset the environment."""
        self.target_state = random.choice(self.states)  # Hidden target port
        self.reward_available = False  # Water is not initially available
        self.step_count = 0
        self.water_available_step = random.randint(10, 30)  # Random moment water becomes available

        if not keep_belief:
            self.belief_state = np.ones(self.num_ports) / self.num_ports  # Reset belief to uniform

    def observe(self, action):
        """Generate an observation based on the hidden state."""
        noisy_distance = abs(self.target_state - action) + np.random.normal(0, self.observation_noise)
        # reward_available is 1 only if the action is correct and water is available
        reward_signal = int(self.reward_available and action == self.target_state)
        observation = {
            "distance": int(round(noisy_distance)),  # Noisy distance to the target port
            "reward_available": reward_signal,       # Reward signal only for correct action
        }
        return observation

    def update_belief(self, action, observation):
        """Update the belief state using Bayes' rule."""
        new_belief = np.zeros(self.num_ports)
        for s in self.states:
            prob_distance = self.O["distance"][s, action, observation["distance"]]
            prob_reward = self.O["reward_available"][s, action, observation["reward_available"]]
            
            # Adjust weights for reward signal influence
            combined_prob = prob_distance * (0.8 * prob_reward + 0.2)
            new_belief[s] = self.belief_state[s] * combined_prob * np.sum(self.T[:, action, s])
        
        new_belief /= np.sum(new_belief)  # Normalize
        self.belief_state = new_belief

    def step(self, action):
        """Take a step in the environment."""
        reward = 0
        if self.reward_available and action == self.target_state:
            reward = self.R[self.target_state, action]  # Reward only if correct action and water available

        # Generate observation and update belief state
        observation = self.observe(action)
        self.update_belief(action, observation)

        # Update step count and water availability
        self.step_count += 1
        if self.step_count == self.water_available_step:
            self.reward_available = True  # Water becomes available after this step

        return reward, observation

    def select_action(self):
        """Select an action based on the current belief state."""
        q_values = np.zeros(self.num_ports)
        for a in self.actions:
            for s in self.states:
                q_values[a] += self.belief_state[s] * (
                    self.R[s, a]
                    + self.discount_factor * np.sum(self.T[s, a, :] * self.belief_state)
                )
        return np.argmax(q_values)

    def simulate(self, num_steps=50):
        """Simulate the behavior of the mouse."""
        self.reset()
        rewards = []
        actions = []
        observations = []
        belief_history = []

        for step in range(num_steps):
            action = self.select_action()
            reward, observation = self.step(action)

            # Debugging Outputs
            relative_action = (action - self.target_state) % self.num_ports
            if relative_action > self.num_ports // 2:
                relative_action -= self.num_ports
            print(f"Step {step}: Action={action}, Relative Action={relative_action}, Reward={reward}, Observation={observation}")
            print(f"Belief State: {self.belief_state}")

            rewards.append(reward)
            actions.append(action)
            observations.append(observation)
            belief_history.append(self.belief_state.copy())

        return rewards, actions, observations, belief_history

    def plot_behavior(self, rewards, actions, observations, belief_history):
        """Plot the behavior of the mouse."""
        cumulative_rewards = np.cumsum(rewards)
        water_available = [obs["reward_available"] for obs in observations]

        plt.figure(figsize=(14, 12))

        # Cumulative Rewards
        plt.subplot(4, 2, 1)
        plt.plot(cumulative_rewards, label="Cumulative Rewards")
        
        plt.axvspan(self.water_available_step, len(actions), color='lightgreen', alpha=0.3, label="Water Available")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative Reward")
        plt.title("Reward Collection Over Time")
        plt.legend()

        # Actions with Water Availability Zone
        plt.subplot(4, 2, 2)
        plt.plot(actions, 'o-', label="Actions (Ports)")
        plt.axvspan(self.water_available_step, len(actions), color='lightgreen', alpha=0.3, label="Water Available")
        plt.xlabel("Steps")
        plt.ylabel("Action")
        plt.title("Actions and Water Availability")
        plt.legend()

        # Belief States Over Time
        plt.subplot(4, 2, 3)
        for port in range(self.num_ports):
            belief_values = [belief[port] for belief in belief_history]
            plt.plot(belief_values, label=f"Belief for Port {port}")
        
        plt.axvspan(self.water_available_step, len(actions), color='lightgreen', alpha=0.3, label="Water Available")
        plt.xlabel("Steps")
        plt.ylabel("Belief")
        plt.title("Belief State Evolution")
        plt.legend()

        # Distance Observations
        distances = [obs["distance"] for obs in observations]
        plt.subplot(4, 2, 4)
        plt.plot(distances, 'o-', label="Distance Observed")
        
        plt.axvspan(self.water_available_step, len(actions), color='lightgreen', alpha=0.3, label="Water Available")
        plt.xlabel("Steps")
        plt.ylabel("Distance")
        plt.title("Distance Observations")
        plt.legend()

        # Histogram of Selected Ports
        relative_ports = [(action - self.target_state) % self.num_ports for action in actions]
        relative_ports = [port - self.num_ports if port > self.num_ports // 2 else port for port in relative_ports]
        plt.subplot(4, 2, 5)
        plt.hist(relative_ports, bins=np.arange(-4.5, 5, 1), align="mid", rwidth=0.8)
        plt.xlabel("Relative Port (Correct Port = 0)")
        plt.ylabel("Frequency")
        plt.title("Histogram of Selected Ports")

        plt.tight_layout()
        plt.show()

        # Print the correct port
        print(f"Correct Port: {self.target_state}")

