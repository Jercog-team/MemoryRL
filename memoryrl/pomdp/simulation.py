# pomdp/simulation.py
from typing import Any, Dict, List

import numpy as np
from numpy.random import default_rng

from .belief import initialize_belief
from .policy import epsilon_greedy_policy


class MemoryAgent:
    """
    Agent using your age×prior heuristic belief update + epsilon-greedy policy.

    Notes
    -----
    - prior_ports : f_0(r) from initialize_belief (static per episode).
    - last_visit_trials : T_r updated after each poke.
    - At each trial we call update_belief(prior_ports, T_r, t) to get f_t(r),
      then add distance penalty and epsilon-greedy.
    """

    def __init__(
        self,
        k_today,
        ky_vector,
        w_vector,
        eps_first,
        eps_rest,
        lambda_dist,
        port_angles,
        rng,
    ):
        self.k_today = k_today
        self.ky_vector = np.array(ky_vector, dtype=float)
        self.w_vector = np.array(w_vector, dtype=float)
        self.eps_first = float(eps_first)
        self.eps_rest = float(eps_rest)
        self.lambda_dist = float(lambda_dist)
        self.port_angles = np.asarray(port_angles, dtype=float)
        self.rng = rng

        self.prior_ports = None        # f_0(r)
        self.last_visit_trials = None  # T_r
        self.belief0 = None            # P(r, τ) from initialize_belief (if you want to keep it)

    def reset_episode(
        self,
        episode_idx,
        ports_true_all,
        lag_ports_true_all,
        distance_bin,
    ):
        """
        Initialize belief for a given episode using initialize_belief + per-distance
        k_yesterday and w_today (like your notebook).
        """
        k_yesterday = float(self.ky_vector[distance_bin])
        w_today = float(self.w_vector[distance_bin])

        belief0, f_theta0, _ = initialize_belief(
            episode_idx,
            ports_true_all,
            lag_ports_true_all,
            self.k_today,
            k_yesterday,
            w_today,
        )

        self.belief0 = belief0
        self.prior_ports = f_theta0
        self.last_visit_trials = np.full(8, -1, dtype=int)  # T_r_sim

    def select_action(
        self,
        trial_index,
        step_in_trial,
        r_cur,
        theta_cur,
    ):
        """
        Choose a port index in 0..7 using the heuristic update_belief + epsilon-greedy.
        """
        return epsilon_greedy_policy(
            prior_ports=self.prior_ports,
            last_visit_trials=self.last_visit_trials,
            trial_index=trial_index,
            r_cur=r_cur,
            theta_cur=theta_cur,
            port_angles=self.port_angles,
            lambda_dist=self.lambda_dist,
            eps_first=self.eps_first,
            eps_rest=self.eps_rest,
            step_in_trial=step_in_trial,
            rng=self.rng,
        )

    def observe(self, action_port_idx, reward, trial_index):
        """
        Update internal agent state after action and reward.

        Right now:
          - we update last_visit_trials (T_r)
          - we DO NOT modify prior_ports, so behaviour matches your notebook.
        """
        self.last_visit_trials[action_port_idx] = trial_index
        # reward currently not used for belief in this heuristic version.


def simulate_sessions(
    session_order: List[int],
    distance_seq_oversampled: np.ndarray,
    arr_ports_oversampled: np.ndarray,
    ports_true_all: np.ndarray,
    lag_ports_true_all: np.ndarray,
    positionsStartTrial_polar_oversampled: List[np.ndarray],
    data_pokes_masked: Dict[str, List[List[np.ndarray]]],
    tau_true_per_session: np.ndarray,
    k_today: float,
    ky_vector: List[float],
    w_vector: List[float],
    eps_first: float,
    eps_rest: float,
    lambda_dist: float,
    max_pokes_per_trial: int,
    port_angles: np.ndarray,
    rng_seed: int = 123,
) -> Dict[str, Any]:
    """
    Run the POMDP simulation over oversampled sessions.

    This is the structured version of your notebook loop, but the 'belief'
    part is now explicit via update_belief. Behaviour is identical.

    Returns
    -------
    dict with keys:
      - sim_pokes_sessions_pen        : list[sessions][trials][ports(1..8)]
      - hist_per_trial_sessions_pen   : list[sessions](T,8)
      - hist_per_session_pen          : list[sessions](8,)
      - tau_true_per_session_out      : list[int]
      - total_pokes                   : int
    """
    rng = default_rng(rng_seed)

    distance_seq_oversampled = np.asarray(distance_seq_oversampled, dtype=int)
    arr_ports_oversampled = np.asarray(arr_ports_oversampled, dtype=int)
    ports_true_all = np.asarray(ports_true_all, dtype=int)
    lag_ports_true_all = np.asarray(lag_ports_true_all, dtype=int)
    tau_true_per_session = np.asarray(tau_true_per_session, dtype=int)
    port_angles = np.asarray(port_angles, dtype=float)

    agent = MemoryAgent(
        k_today=k_today,
        ky_vector=ky_vector,
        w_vector=w_vector,
        eps_first=eps_first,
        eps_rest=eps_rest,
        lambda_dist=lambda_dist,
        port_angles=port_angles,
        rng=rng,
    )

    sim_pokes_sessions_pen: List[Any] = []
    hist_per_trial_sessions_pen: List[np.ndarray] = []
    hist_per_session_pen: List[np.ndarray] = []
    tau_true_list: List[int] = []
    total_pokes = 0

    for i in range(len(session_order)):
        current_distance = int(distance_seq_oversampled[i])
        current_arr_port = int(arr_ports_oversampled[i])  # correct port (1..8)
        original_session_index = int(session_order[i])
        current_positions_start = positionsStartTrial_polar_oversampled[i]

        T_max_ep = len(data_pokes_masked["all_pokes"][i])
        tau_true = int(tau_true_per_session[i])
        tau_true_list.append(tau_true)
        reward_flag = 0

        agent.reset_episode(
            episode_idx=original_session_index,
            ports_true_all=ports_true_all,
            lag_ports_true_all=lag_ports_true_all,
            distance_bin=current_distance,
        )

        session_trials: List[List[int]] = []
        session_hist = np.zeros(8, dtype=int)
        trial_hist_list: List[np.ndarray] = []

        start_polar = np.asarray(current_positions_start, dtype=float)

        for t in range(T_max_ep):  # trial loop
            r_cur = float(start_polar[t, 0])
            theta_cur = float(start_polar[t, 1])

            trial_pokes: List[int] = []
            trial_hist = np.zeros(8, dtype=int)

            n_pokes_this_trial = len(data_pokes_masked["all_pokes"][i][t])
            n_pokes_this_trial = min(n_pokes_this_trial, max_pokes_per_trial)

            for s in range(n_pokes_this_trial):
                # action in 0..7
                p = agent.select_action(
                    trial_index=t,
                    step_in_trial=s,
                    r_cur=r_cur,
                    theta_cur=theta_cur,
                )

                # observe no reward by default
                agent.observe(action_port_idx=p, reward=0, trial_index=t)

                trial_pokes.append(p)
                trial_hist[p] += 1
                session_hist[p] += 1

                # reward condition identical to your code
                if (p == current_arr_port - 1) and (t >= tau_true):
                    reward_flag = 1
                    agent.observe(action_port_idx=p, reward=1, trial_index=t)
                    r_cur = 1.0
                    theta_cur = port_angles[p]
                    break
                else:
                    r_cur = 1.0
                    theta_cur = port_angles[p]

            session_trials.append(trial_pokes)
            trial_hist_list.append(trial_hist)

            if reward_flag == 1:
                break

        if not trial_hist_list:
            trial_hist_list.append(np.zeros(8, dtype=int))

        sim_pokes_sessions_pen.append(session_trials)
        hist_per_trial_sessions_pen.append(np.vstack(trial_hist_list))
        hist_per_session_pen.append(session_hist)
        total_pokes += sum(len(tr) for tr in session_trials)

    # convert to 1..8 like in your code
    sim_pokes_sessions_pen = [
        [[int(a) + 1 for a in trial] for trial in session]
        for session in sim_pokes_sessions_pen
    ]

    return {
        "sim_pokes_sessions_pen": sim_pokes_sessions_pen,
        "hist_per_trial_sessions_pen": hist_per_trial_sessions_pen,
        "hist_per_session_pen": hist_per_session_pen,
        "tau_true_per_session": tau_true_list,
        "total_pokes": total_pokes,
    }
