"""
This module defines the DQNLSTM neural network and the DQNAgent class for reinforcement learning.
The DQNLSTM is a deep Q-network with an LSTM layer for handling sequential data.
The DQNAgent class implements the agent's behavior, including action selection, experience storage, and training.
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class DQNLSTM(nn.Module):
    """
    A Deep Q-Network with an LSTM layer for handling sequential data.

    Attributes:
        hidden_dim (int): The number of hidden units in the LSTM layer.
        fc1 (nn.Linear): The first fully connected layer.
        lstm (nn.LSTM): The LSTM layer.
        fc2 (nn.Linear): The second fully connected layer for output.
    """

    def __init__(self, input_dim, action_dim, hidden_dim=128):
        """
        Initialize the DQNLSTM network.

        Args:
            input_dim (int): The dimension of the input features.
            action_dim (int): The number of possible actions.
            hidden_dim (int, optional): The number of hidden units in the LSTM layer. Default is 128.
        """
        super(DQNLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.lstm = nn.LSTM(128, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x, hidden):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor.
            hidden (tuple): The hidden state of the LSTM.

        Returns:
            torch.Tensor: The output Q-values for each action.
            tuple: The updated hidden state of the LSTM.
        """
        x = torch.relu(self.fc1(x))
        x, hidden = self.lstm(x.unsqueeze(0), hidden)
        x = self.fc2(x.squeeze(0))
        
        #noise = torch.randn_like(x) * 0.05  # Small noise to prevent deterministic behavior
        return x, hidden

    def init_hidden(self):
        """
        Initialize the hidden state of the LSTM.

        Returns:
            tuple: A tuple containing the initial hidden state and cell state of the LSTM.
        """
        return (torch.zeros(1, 1, self.hidden_dim), torch.zeros(1, 1, self.hidden_dim))

class DQNAgent:
    """
    A Deep Q-Network (DQN) agent with an LSTM-based policy network.

    Attributes:
        action_dim (int): The number of possible actions.
        gamma (float): The discount factor for future rewards.
        batch_size (int): The size of the training batch.
        memory (deque): A replay memory to store experiences.
        policy_net (DQNLSTM): The policy network.
        target_net (DQNLSTM): The target network.
        optimizer (torch.optim.Optimizer): The optimizer for training the policy network.
        loss_fn (torch.nn.Module): The loss function for training.
    """

    def __init__(self, input_dim, action_dim, gamma=0.99, lr=1e-3, batch_size=16):
        """
        Initialize the DQNAgent.

        Args:
            input_dim (int): The dimension of the input features.
            action_dim (int): The number of possible actions.
            gamma (float, optional): The discount factor for future rewards. Default is 0.99.
            lr (float, optional): The learning rate for the optimizer. Default is 1e-3.
            batch_size (int, optional): The size of the training batch. Default is 16.
        """
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = deque(maxlen=1000)
        
        self.policy_net = DQNLSTM(input_dim, action_dim)
        self.target_net = DQNLSTM(input_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
    
    def to(self, device):
        """
        Move the agent's networks to the specified device.

        Args:
            device (torch.device): The device to move the networks to (e.g., 'cpu' or 'cuda').
        """
        self.policy_net = self.policy_net.to(device)
        self.target_net = self.target_net.to(device)

    def select_action(self, state, hidden, last_action, trial_first_poke, epsilon=0.2, distance_bias=1, repeat_penalty=5.0):
        """
        Select an action with distance bias and penalty for repeating the same port within a trial.

        Args:
            state (torch.Tensor): The current state tensor.
            hidden (tuple): The hidden state of the LSTM.
            last_action (int): Last poked port (only used if NOT the first poke of a trial).
            trial_first_poke (bool): True if this is the first poke of the trial.
            epsilon (float, optional): Exploration probability. Default is 0.2.
            distance_bias (float, optional): Strength of distance-based preference. Default is 1.
            repeat_penalty (float, optional): Penalty for repeating the same port. Default is 5.0.

        Returns:
            int: Selected action (port to poke next).
            tuple: Updated LSTM hidden state.
        """
        
        num_ports = self.action_dim  # Assuming 8 ports
        device = next(self.policy_net.parameters()).device
        if random.random() < epsilon:
            action = random.randint(0, num_ports - 1)
            return action, hidden

        with torch.no_grad():
            q_values, hidden = self.policy_net(state, hidden)

            if trial_first_poke:
                # No penalties for first poke
                action = torch.argmax(q_values).item()
            else:
                distances = torch.tensor([
                    min(abs(a - last_action), num_ports - abs(a - last_action)) 
                    for a in range(num_ports)
                ], dtype=torch.float32).to(device)

                distance_weights = torch.exp(-distance_bias * distances)
                weighted_q_values = q_values * distance_weights

                # Apply a penalty to the previously poked port
                penalty_mask = torch.zeros_like(q_values)
                penalty_mask[0, last_action] = -repeat_penalty

                final_q_values = weighted_q_values + penalty_mask
                probs = torch.softmax(final_q_values / 0.5, dim=-1)
                action = torch.multinomial(probs, 1).item()

            return action, hidden
        

    def store_experience(self, experience):
        """
        Store an experience in the replay memory.

        Args:
            experience (tuple): A tuple containing (state, action, reward, next_state, done, hidden).
        """
        self.memory.append(experience)


    def train(self):
        """
        Train the policy network using experiences from the replay memory.

        Returns:
            None
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample a batch of experiences
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones, hiddens = zip(*batch)

        # Move tensors to the same device as the policy network
        device = next(self.policy_net.parameters()).device
        states = torch.cat(states).to(device)
        next_states = torch.cat(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        # Compute Q-values for the current states
        q_values, _ = self.policy_net(states, hiddens[0])
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        # Compute target Q-values for the next states
        with torch.no_grad():
            next_q_values, _ = self.target_net(next_states, hiddens[0])
            max_next_q_values = next_q_values.max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        # Compute loss and update the policy network
        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def update_target(self):
        """
        Update the target network by copying the weights from the policy network.

        Returns:
            None
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())