def simulate_POMDP(n_episodes=500, kappa=0.5, w_today=0.3, w_yesterday=0.2, greedy_policy = 'marginal belief', water_availability_dist=True, n_ports=8, n_trials=20):

    arr_ports = []  # Track the correct ports for each episode
    Big_Hist_data = []  # Track the poke distributions for each episode
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}
    poke_sequences_sim = [] # Track the sequences of pokes for each episode
    
    for episode in range(n_episodes):
        b, r_true, tau_true, last_visit, prev_r_true, f_theta = initialize_belief(kappa, w_today=w_today, w_yesterday=w_yesterday, water_availability_dist=water_availability_dist)
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




def simulate_LSTM_POMDP(n_episodes=5000, max_pokes=4, epsilon_decay=0.99, epsilon_min=0.1,pokes_per_trial_dist=trial_sums , water_availability_dist=True, kappa=3, w_today=0.3, w_yesterday=0.2, epsilon=0.2):
    
    agent = DQNAgent(input_dim=n_ports * n_trials + n_ports + 1, action_dim=n_ports)

    Big_Hist_data = []
    poke_sequences_sim = []
    arr_ports = []
    distances_sim = {0: [], 1: [], 2: [], 3: [], 4: []}
    
    for episode in range(n_episodes):
        b_prior, r_true, tau_true, last_visit, prev_r_true = initialize_belief(kappa=kappa, w_today=w_today, w_yesterday=w_yesterday, water_availability_dist=water_availability_dist)
        dist = abs(r_true - prev_r_true) % 5
        poke_distribution = np.zeros(n_ports)
        last_visits = [None] * n_ports
        phase = 0
        done = False
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


        
        Big_Hist_data.append(poke_distribution)
        distances_sim[dist].append(np.roll(poke_distribution, 3 - r_true))
        arr_ports.append(r_true)
        
        agent.update_target()
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    simulated_hist = plot_histogram(Big_Hist_data, arr_ports)


    return simulated_hist, distances_sim, poke_sequences_sim

