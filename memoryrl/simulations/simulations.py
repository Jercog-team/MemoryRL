import numpy as np
import torch
from memoryrl.pomdp.belief import initialize_belief_von_misses, initialize_belief_deltas, initialize_belief_uniform, update_belief
from pomdp.policy import greedy_policy_marginal_belief, greedy_policy_last_visit, bellman_policy
from memoryrl.visualization.plotting import plot_histogram
from memoryrl.utils.preprocessing_functions import circular_distance
from memoryrl.agents.agents import DQNAgent
import pickle
import importlib.resources as pkg_resources
from distributions import __name__ as dist_pkg
import math


def simulate_POMDP(n_episodes=500,
                   kappa_today=0.5,
                   kappa_yesterday=0.5,
                   w_today=0.3,
                   w_yesterday=0.2,
                   policy='marginal belief',
                   water_availability_dist=True,
                   pokes_per_trial_dist=None,
                   ports_sequences='real',
                   n_ports=8,
                   n_trials=20,
                   always_reward=False,
                   belief_init='von misses'):

    """
    Simulates a Partially Observable Markov Decision Process (POMDP) using a greedy policy.

    Parameters:
        n_episodes (int): Number of episodes to simulate.
        kappa (float): Parameter for belief initialization.
        w_today (float): Weight for today's belief update.
        w_yesterday (float): Weight for yesterday's belief update.
        policy (str): Policy type ('marginal belief', 'last visit' or 'bellman').
        water_availability_dist (bool): Whether to use water availability distribution.
        pokes_per_trial_dist (list): Distribution of pokes per trial.
        ports_sequences (str): Specifies the source of the port sequences for the simulation.
            - 'real': Use preloaded real-world port sequences from a dataset.
            - 'random': Generate random port sequences for the simulation.
        n_ports (int): Number of ports in the environment.
        n_trials (int): Number of trials per episode.
        belief_init (str): Method for initializing the belief state. 
            Options are 'von misses' , 'deltas' or 'uniform'.

    Returns:
        tuple: Simulated histogram, distances simulation, and poke sequences simulation.
    """

    new_AngRad = {1: -3*np.pi/4, 2: -np.pi/2, 3: -np.pi/4, 4: 0, 5: np.pi/4, 6: np.pi/2, 7: 3*np.pi/4, 8: np.pi}
    arr_ports = []  # Track the correct ports for each episode
    Big_Hist_data = []  # Track the poke distributions for each episode
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}
    poke_sequences_sim = [] # Track the sequences of pokes for each episode
    animation_data = [] # Track the animation data for each episode
    diffs = []
    taus = []


    if ports_sequences == 'real':
        ports_seqs = np.load(pkg_resources.files(dist_pkg).joinpath('ports.npz'))
        ceil = math.ceil(n_episodes / np.shape(ports_seqs['arr1'])[0])
        ports = np.repeat(ports_seqs['arr1'], ceil)
        lag_ports = np.repeat(ports_seqs['arr2'], ceil)

    elif ports_sequences == 'random':
        ports = np.random.randint(0, n_ports, size=n_episodes)
        lag_ports = np.random.randint(0, n_ports, size=n_episodes)

    else:
        ports = ports_sequences[0]
        lag_ports = ports_sequences[1]

    port_seq_rad=np.vectorize(new_AngRad.get)(np.array(ports))
    lags_port_seq_rad =np.vectorize(new_AngRad.get)(np.array(lag_ports))

    distances = circular_distance(port_seq_rad,lags_port_seq_rad)*8/(2*np.pi)


    if not pokes_per_trial_dist:
        with pkg_resources.files(dist_pkg).joinpath("pokes_per_trial_dist.pkl").open("rb") as f1:
            pokes_per_trial_dist = pickle.load(f1)
    
    for episode in range(n_episodes):

        r_true = ports[episode]
        prev_r_true = lag_ports[episode]

        if belief_init == 'von misses':
            b, tau_true, last_visit, f_theta = initialize_belief_von_misses(r_true, prev_r_true, kappa_today=kappa_today, kappa_yesterday=kappa_yesterday, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        elif belief_init == 'deltas':   
            b, tau_true, last_visit, f_theta = initialize_belief_deltas(r_true, prev_r_true, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        elif belief_init == 'uniform':
            b, tau_true, last_visit, f_theta = initialize_belief_uniform( water_availability_dist=water_availability_dist)
        
        dist = distances[episode]  # CHECK THIS
        poke_distribution = np.zeros(n_ports)
        done = False
        
        if not always_reward:
            for t in range(n_trials):
                poke_sequences_sim_trial = []
                max_pokes = np.random.choice(pokes_per_trial_dist)
                for _ in range(max_pokes):
                    if policy == 'marginal belief':
                        action, _ = greedy_policy_marginal_belief(b)

                    elif policy == 'last visit':
                        action, _ = greedy_policy_last_visit(f_theta, last_visit, t)

                    elif policy == 'bellman':
                        state = [poke - 1 for poke in poke_sequences_sim_trial]
                        action, _ = bellman_policy(state, b, last_visit, t, n_trials=n_trials)


                    poke_sequences_sim_trial.append(action+1)
                    animation_data.append({
                        "trial": t,
                        "Belief": b.copy(),
                        "chosen_action": action+1})


                    if t<=tau_true:
                        poke_distribution[action] += 1
                    w = int((action == r_true -1) and (t >= tau_true))
                    b, _, last_visit = update_belief(b, w, action, t, last_visit)
                    if w:
                        done = True
                        break
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break
        else:
            t = 1
            while True:
                poke_sequences_sim_trial = []
                max_pokes = np.random.choice(pokes_per_trial_dist)
                for _ in range(max_pokes):
                    if policy == 'marginal belief':
                        action, _ = greedy_policy_marginal_belief(b)

                    elif policy == 'last visit':
                        action, _ = greedy_policy_last_visit(f_theta, last_visit, t)

                    elif policy == 'bellman':
                        state = [poke - 1 for poke in poke_sequences_sim_trial]
                        action, _ = bellman_policy(state, b, last_visit, t, n_trials=n_trials)


                    poke_sequences_sim_trial.append(action+1)
                    poke_distribution[action] += 1
                    animation_data.append({
                        "trial": t,
                        "Belief": b.copy(),
                        "chosen_action": action+1})


                    w = int((action == r_true -1) and (t >= tau_true))

                    b, _, last_visit = update_belief(b, w, action, t, last_visit)
                    t_now = t
                    t += 1
                    if w:
                        diff = t_now - tau_true
                        diffs.append(diff)
                        taus.append(tau_true)
                        done = True
                        break
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break

        Big_Hist_data.append(poke_distribution)
        arr_ports.append(r_true)
        distances_sim[dist].append(np.roll(poke_distribution, 3 - r_true))

    simulated_hist = plot_histogram(Big_Hist_data, arr_ports)

    
    return simulated_hist, distances_sim, poke_sequences_sim, Big_Hist_data, animation_data, diffs, taus


def simulate_LSTM_POMDP(n_episodes=5000,
                        max_pokes=4,
                        epsilon_decay=0.99,
                        epsilon_min=0.1,
                        pokes_per_trial_dist=None,
                        water_availability_dist=True,
                        ports_sequences='real',
                        kappa_today=3,
                        kappa_yesterday=3,
                        w_today=0.3,
                        w_yesterday=0.2,
                        epsilon=0.2,
                        n_ports=8,
                        n_trials=20,
                        always_reward=False,
                        belief_init='von misses',
                        pretrained_agent=None):

    """
    Simulates a POMDP using an LSTM-based Deep Q-Network (DQN) agent.

    Parameters:
        n_episodes (int): Number of episodes to simulate.
        max_pokes (int): Maximum number of pokes per trial.
        epsilon_decay (float): Decay rate for epsilon in epsilon-greedy policy.
        epsilon_min (float): Minimum value for epsilon.
        pokes_per_trial_dist (list): Distribution of pokes per trial.
        water_availability_dist (bool): Whether to use water availability distribution.
        ports_sequences (str): Specifies the source of the port sequences for the simulation.
            - 'real': Use preloaded real-world port sequences from a dataset.
            - 'random': Generate random port sequences for the simulation.
        kappa (float): Parameter for belief initialization.
        w_today (float): Weight for today's belief update.
        w_yesterday (float): Weight for yesterday's belief update.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        n_ports (int): Number of ports in the environment.
        n_trials (int): Number of trials per episode.
        always_reward (bool): If True, the session doesn't end until the agent receives the reward.
        belief_init (str): Method for initializing the belief state. 
            Options are 'von misses' or 'deltas'.
        pretrained_agent (DQNAgent or None): Pretrained agent to use for the simulation. 
            If None, a new agent is initialized.

    Returns:
        tuple: Simulated histogram, distances simulation, and poke sequences simulation.
    """
    agent = pretrained_agent if pretrained_agent is not None else DQNAgent(input_dim=n_ports * n_trials + n_ports + 1, action_dim=n_ports)

    new_AngRad = {1: -3*np.pi/4, 2: -np.pi/2, 3: -np.pi/4, 4: 0, 5: np.pi/4, 6: np.pi/2, 7: 3*np.pi/4, 8: np.pi}
    Big_Hist_data = []
    poke_sequences_sim = []
    arr_ports = []
    animation_data = []
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}
    diffs = []
    taus = []

    if ports_sequences == 'real':
        ports_seqs = np.load(pkg_resources.files(dist_pkg).joinpath('ports.npz'))
        ceil = math.ceil(n_episodes / np.shape(ports_seqs['arr1'])[0])
        ports = np.repeat(ports_seqs['arr1'], ceil)
        lag_ports = np.repeat(ports_seqs['arr2'], ceil)

    elif ports_sequences == 'random':
        ports = np.random.randint(0, n_ports, size=n_episodes)
        lag_ports = np.random.randint(0, n_ports, size=n_episodes)

    else:
        ports = ports_sequences[0]
        lag_ports = ports_sequences[1]

    port_seq_rad=np.vectorize(new_AngRad.get)(np.array(ports))
    lags_port_seq_rad =np.vectorize(new_AngRad.get)(np.array(lag_ports))

    distances = circular_distance(port_seq_rad,lags_port_seq_rad)*8/(2*np.pi)


    if not pokes_per_trial_dist:
        with pkg_resources.files(dist_pkg).joinpath("pokes_per_trial_dist.pkl").open("rb") as f1:
            pokes_per_trial_dist = pickle.load(f1)
    
    for episode in range(n_episodes):

        r_true = ports[episode]
        prev_r_true = lag_ports[episode]

        if belief_init == 'von misses':
            b_prior, tau_true, last_visit, _ = initialize_belief_von_misses(r_true, prev_r_true, kappa_today=kappa_today, kappa_yesterday=kappa_yesterday, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        elif belief_init == 'deltas':   
            b_prior, tau_true, last_visit, _ = initialize_belief_deltas(r_true, prev_r_true, w_today=w_today, w_yesterday=w_yesterday, n_ports=n_ports, n_trials=n_trials, water_availability_dist=water_availability_dist)
        elif belief_init == 'uniform':
            b_prior, tau_true, last_visit, _ = initialize_belief_uniform( water_availability_dist=water_availability_dist)
        
        dist = distances[episode]
        poke_distribution = np.zeros(n_ports)
        done = False

        if not always_reward:

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
                    action, hidden, q_values_before, q_values_after, penalty_mask = agent.select_action(torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), hidden,last_action, trial_first_poke,  epsilon)
                    trial_first_poke = False  # Next pokes within trial follow distance bias
                    last_action = action  # Update last action for next step
                    poke_sequences_sim_trial.append(action+1)

                    animation_data.append({
                    "trial": t,
                    "poke": s,
                    "Belief": b_prior.sum(axis=1).copy(),
                    "q_value_before": q_values_before.clone(),
                    "q_value_after": q_values_after.clone(),
                    "penalty_mask": penalty_mask.clone(),
                    "chosen_action": action})

                    if t<=tau_true:
                        poke_distribution[action-1] += 1
                    w = int((action == r_true) and (t >= tau_true))
                    b_prior, term_prob, last_visit = update_belief(b_prior, w, action, t, last_visit)
                    agent.store_experience((torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), action, w, torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), int(t >= tau_true), hidden))
                    agent.train()

                    if w:
                        done = True
                        break
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break

        else:

            t = 1
            t_now = t
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
                    

                    action, hidden, q_values_before, q_values_after, penalty_mask = agent.select_action(torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), hidden,last_action, trial_first_poke,  epsilon)
                    trial_first_poke = False  # Next pokes within trial follow distance bias
                    last_action = action  # Update last action for next step
                    poke_sequences_sim_trial.append(action)
                    poke_distribution[action] += 1

                    animation_data.append({
                    "trial": t,
                    "poke": s,
                    "Belief": b_prior.sum(axis=1).copy(),
                    "q_value_before": q_values_before.clone(),
                    "q_value_after": q_values_after.clone(),
                    "penalty_mask": penalty_mask.clone(),
                    "chosen_action": action})

                    w = int((action == r_true -1) and (t >= tau_true))

                    b_prior, term_prob, last_visit = update_belief(b_prior, w, action, t, last_visit)
                    agent.store_experience((torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), action, w, torch.tensor(state_vector, dtype=torch.float32).unsqueeze(0), int(t >= tau_true), hidden))
                    agent.train()
                    t_now = t
                    t +=1
                    if w:

                        done = True
                        break

                diff = t_now - tau_true
                diffs.append(diff)
                taus.append(tau_true)
                poke_sequences_sim.append(poke_sequences_sim_trial)
                if done:
                    break


        
        Big_Hist_data.append(poke_distribution)
        distances_sim[dist].append(np.roll(poke_distribution, 4 - r_true))
        arr_ports.append(r_true)
        
        agent.update_target()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    simulated_hist = plot_histogram(Big_Hist_data, arr_ports)


    return simulated_hist, distances_sim, poke_sequences_sim, agent, Big_Hist_data, animation_data, diffs, taus

