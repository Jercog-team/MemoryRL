import numpy as np
import random
import torch
from memoryrl.pomdp.belief import initialize_belief_von_misses, initialize_belief_deltas, update_belief
from memoryrl.pomdp.greedy_policy import greedy_policy_marginal_belief, greedy_policy_last_visit
from memoryrl.visualization.plotting import plot_histogram
from memoryrl.agents.agents import DQNAgent
import pickle
import importlib.resources as pkg_resources
from distributions import __name__ as dist_pkg


def simulate_POMDP(n_episodes=500,
                   kappa=0.5,
                   w_today=0.3,
                   w_yesterday=0.2,
                   greedy_policy='marginal belief',
                   water_availability_dist=True,
                   n_ports=8,
                   n_trials=20,
                   belief_init='von misses'):
    """
    Simulates a Partially Observable Markov Decision Process (POMDP) using a greedy policy.

    Parameters:
        n_episodes (int): Number of episodes to simulate.
        kappa (float): Parameter for belief initialization.
        w_today (float): Weight for today's belief update.
        w_yesterday (float): Weight for yesterday's belief update.
        greedy_policy (str): Policy type ('marginal belief' or 'last visit').
        water_availability_dist (bool): Whether to use water availability distribution.
        n_ports (int): Number of ports in the environment.
        n_trials (int): Number of trials per episode.

    Returns:
        tuple: Simulated histogram, distances simulation, and poke sequences simulation.
    """
    arr_ports = []  # Track the correct ports for each episode
    Big_Hist_data = []  # Track the poke distributions for each episode
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}
    poke_sequences_sim = [] # Track the sequences of pokes for each episode
    
    for episode in range(n_episodes):
        
        if belief_init == 'von misses':
            b, r_true, tau_true, last_visit, prev_r_true, f_theta = initialize_belief_von_misses(kappa=kappa, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        else:   
            b, r_true, tau_true, last_visit, prev_r_true, f_theta = initialize_belief_deltas( w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        
        dist = abs(r_true - prev_r_true) % 5
        poke_distribution = np.zeros(n_ports)
        done = False
        
        for t in range(n_trials):
            poke_sequences_sim_trial = []
            max_pokes = random.randint(1, 4)
            for _ in range(max_pokes):
                if greedy_policy == 'marginal belief':
                    action, _ = greedy_policy_marginal_belief(b)

                elif greedy_policy == 'last visit':
                    action, _ = greedy_policy_last_visit(f_theta, last_visit, t)

                poke_sequences_sim_trial.append(action+1)
                poke_distribution[action] += 1
                w = int((action == r_true) and (t >= tau_true))
                b, _, last_visit = update_belief(b, w, action, t, last_visit)
                if w:
                    done = True
                    break
            poke_sequences_sim.append(poke_sequences_sim_trial)
            if done:
                break
        Big_Hist_data.append(poke_distribution)
        arr_ports.append(r_true)
        distances_sim[dist].append(np.roll(poke_distribution, 3 - r_true))

    simulated_hist = plot_histogram(Big_Hist_data, arr_ports)

    
    return simulated_hist, distances_sim, poke_sequences_sim


def simulate_LSTM_POMDP(n_episodes=5000,
                        max_pokes=4,
                        epsilon_decay=0.99,
                        epsilon_min=0.1,
                        pokes_per_trial_dist=None,
                        water_availability_dist=True,
                        kappa=3,
                        w_today=0.3,
                        w_yesterday=0.2,
                        epsilon=0.2,
                        n_ports=8,
                        n_trials=20,
                        belief_init='von misses'):
    """
    Simulates a POMDP using an LSTM-based Deep Q-Network (DQN) agent.

    Parameters:
        n_episodes (int): Number of episodes to simulate.
        max_pokes (int): Maximum number of pokes per trial.
        epsilon_decay (float): Decay rate for epsilon in epsilon-greedy policy.
        epsilon_min (float): Minimum value for epsilon.
        pokes_per_trial_dist (list): Distribution of pokes per trial.
        water_availability_dist (bool): Whether to use water availability distribution.
        kappa (float): Parameter for belief initialization.
        w_today (float): Weight for today's belief update.
        w_yesterday (float): Weight for yesterday's belief update.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        n_ports (int): Number of ports in the environment.
        n_trials (int): Number of trials per episode.

    Returns:
        tuple: Simulated histogram, distances simulation, and poke sequences simulation.
    """
    agent = DQNAgent(input_dim=n_ports * n_trials + n_ports + 1, action_dim=n_ports)

    Big_Hist_data = []
    poke_sequences_sim = []
    arr_ports = []
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}

    if not pokes_per_trial_dist:
        with pkg_resources.files(dist_pkg).joinpath("pokes_per_trial_dist.pkl").open("rb") as f1:
            pokes_per_trial_dist = pickle.load(f1)
    
    for episode in range(n_episodes):

        if belief_init == 'von misses':
            b_prior, r_true, tau_true, last_visit, prev_r_true, _ = initialize_belief_von_misses(kappa=kappa, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        else:   
            b_prior, r_true, tau_true, last_visit, prev_r_true, _ = initialize_belief_deltas( w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        
        dist = abs(r_true - prev_r_true) % 5
        poke_distribution = np.zeros(n_ports)
        done = False

        if n_trials:

            for t in range(n_trials):
                poke_sequences_sim_trial = []
                max_pokes = np.random.choice(pokes_per_trial_dist)
                trial_first_poke = True  # Reset for each trial
                last_action = None
                for s in range(max_pokes):
                    hidden = agent.policy_net.init_hidden()

                    state_vector = np.concatenate([
                                    b_prior.flatten(),       # flattened belief state (160 elements)
                                    last_visit / n_trials,   # normalized last visit array (8 elements)
                                    np.array([t / n_trials]) # normalized current trial scalar (1 element)
                                ])
                    action, hidden = agent.select_action(torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), hidden,last_action, trial_first_poke,  epsilon)
                    trial_first_poke = False  # Next pokes within trial follow distance bias
                    last_action = action  # Update last action for next step
                    poke_sequences_sim_trial.append(action+1)
                    poke_distribution[action] += 1
                    w = int((action == r_true) and (t + 1 >= tau_true))
                    b_prior, term_prob, last_visit = update_belief(b_prior, w, action, t, last_visit)
                    agent.store_experience((torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), action, w, torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), int(t + 1 >= tau_true), hidden))
                    agent.train()
                    if w:
                        done = True
                        break
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break

        else:

            t = 0
            while True:
                poke_sequences_sim_trial = []
                max_pokes = np.random.choice(pokes_per_trial_dist)
                trial_first_poke = True  # Reset for each trial
                last_action = None
                for s in range(max_pokes):
                    hidden = agent.policy_net.init_hidden()


                    if last_visit.sum() > 0:
                        normalized_last_visit = last_visit / last_visit.sum()
                    else:
                        normalized_last_visit = np.zeros_like(last_visit)

                    normalized_t = np.array([t / max_pokes])



                    state_vector = np.concatenate([
                                    b_prior.flatten(),       # flattened belief state (160 elements)
                                    normalized_last_visit,   # normalized last visit array (8 elements)
                                    normalized_t # normalized current trial scalar (1 element)
                                ])
                    

                    action, hidden = agent.select_action(torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), hidden,last_action, trial_first_poke,  epsilon)
                    trial_first_poke = False  # Next pokes within trial follow distance bias
                    last_action = action  # Update last action for next step
                    poke_sequences_sim_trial.append(action+1)
                    poke_distribution[action] += 1
                    w = int((action == r_true) and (t + 1 >= tau_true))
                    b_prior, term_prob, last_visit = update_belief(b_prior, w, action, t, last_visit)
                    agent.store_experience((torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), action, w, torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), int(t + 1 >= tau_true), hidden))
                    agent.train()
                    t +=1
                    if w:
                        done = True
                        break
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break


        
        Big_Hist_data.append(poke_distribution)
        distances_sim[dist].append(np.roll(poke_distribution, 3 - r_true))
        arr_ports.append(r_true)
        
        agent.update_target()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    simulated_hist = plot_histogram(Big_Hist_data, arr_ports)


    return simulated_hist, distances_sim, poke_sequences_sim

