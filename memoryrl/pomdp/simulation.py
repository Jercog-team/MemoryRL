# pomdp/simulation.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.random import default_rng, Generator

from .belief import initialize_belief, update_belief
from .policy import epsilon_greedy_policy
from .agents.dqn_recurrent import DQNRecurrentAgent
from .irl.gail_recurrent import GAILRecurrent


class MemoryAgent:
    """
    POMDP agent with pluggable policy:

    policy_type:
      - "heuristic" : your age×prior memory policy (default, identical to notebook)
      - "dqn"       : recurrent DQN policy (Q-network)
      - "gail"      : recurrent stochastic policy from GAIL

    The environment only calls:
      - reset_episode(...)
      - select_action(...)
      - observe(...)
    """

    def __init__(
        self,
        k_today: float,
        ky_vector: List[float],
        w_vector: List[float],
        eps_first: float,
        eps_rest: float,
        lambda_dist: float,
        port_angles: np.ndarray,
        rng: Generator,
        policy_type: Literal["heuristic", "dqn", "gail"] = "heuristic",
        dqn_agent: Optional[DQNRecurrentAgent] = None,
        gail: Optional[GAILRecurrent] = None,
        max_trials: int = 20,
    ):
        self.k_today = float(k_today)
        self.ky_vector = np.array(ky_vector, dtype=float)
        self.w_vector = np.array(w_vector, dtype=float)
        self.eps_first = float(eps_first)
        self.eps_rest = float(eps_rest)
        self.lambda_dist = float(lambda_dist)
        self.port_angles = np.asarray(port_angles, dtype=float)
        self.rng = rng

        # Belief-related state
        self.prior_ports: np.ndarray | None = None     # f_0(r) from initialize_belief
        self.last_visit_trials: np.ndarray | None = None  # T_r per port
        self.belief0: np.ndarray | None = None         # P(r, τ) (optional reference)
        self.distance_bin: int | None = None           # distance to yesterday port

        # Policy config
        self.policy_type = policy_type
        self.dqn_agent = dqn_agent
        self.gail = gail
        self.max_trials = int(max_trials)

        # Recurrent hidden state for DQN / GAIL
        self.dqn_hidden = None
        self.gail_hidden = None

    # ------------------------------------------------------------------ #
    # Episode init & belief helper                                       #
    # ------------------------------------------------------------------ #

    def reset_episode(
        self,
        episode_idx: int,
        ports_true_all: np.ndarray,
        lag_ports_true_all: np.ndarray,
        distance_bin: int,
    ):
        """
        Initialize belief for a given episode using initialize_belief
        and reset recurrent hidden states.
        """
        k_yesterday = float(self.ky_vector[distance_bin])
        w_today = float(self.w_vector[distance_bin])

        belief0, f_theta0, _ = initialize_belief(
            episode=episode_idx,
            ports_true=ports_true_all,
            lag_ports_true=lag_ports_true_all,
            kappa_today=self.k_today,
            kappa_yesterday=k_yesterday,
            w_today=w_today,
        )

        self.belief0 = belief0
        self.prior_ports = f_theta0              # static f_0(r)
        self.last_visit_trials = np.full(8, -1, dtype=int)  # T_r
        self.distance_bin = int(distance_bin)

        # reset recurrent states
        self.dqn_hidden = None
        self.gail_hidden = None

    def _current_port_belief(self, trial_index: int) -> np.ndarray:
        """
        Belief over reward ports at trial t using your heuristic:

            f_t(r) ∝ f_0(r) * age_t(r)
        """
        return update_belief(
            prior_ports=self.prior_ports,
            last_visit_trials=self.last_visit_trials,
            trial_index=trial_index,
            reward=None,
        )

    def build_obs(
        self,
        trial_index: int,
        step_in_trial: int,
        r_cur: float,
        theta_cur: float,
    ) -> np.ndarray:
        """
        Build observation vector for DQN / GAIL.

        You can tweak this later; for now:

          obs = [
              f_t(r)            (8 dims)   belief over ports
              age_t(r)/Tmax     (8 dims)   normalized age since last visit
              trial_index/Tmax  (1 dim)
              distance_bin/4.   (1 dim)   (assuming distances 0..4)
          ]
          --> obs_dim = 8 + 8 + 1 + 1 = 18
        """
        f_t = self._current_port_belief(trial_index)  # (8,)

        age = (trial_index - self.last_visit_trials).astype(float)
        age[age < 0] = 0.0
        age_norm = age / max(self.max_trials, 1)

        trial_norm = np.array([trial_index / max(self.max_trials, 1)], dtype=float)
        dist_norm = np.array([self.distance_bin / 4.0], dtype=float)

        obs = np.concatenate([f_t, age_norm, trial_norm, dist_norm], axis=0)
        return obs

    # ------------------------------------------------------------------ #
    # Policy selection                                                   #
    # ------------------------------------------------------------------ #

    def select_action(
        self,
        trial_index: int,
        step_in_trial: int,
        r_cur: float,
        theta_cur: float,
    ) -> int:
        """
        Choose a port index in 0..7 based on the selected policy_type.

        This is the ONLY method the environment calls for action selection.
        """

        if self.policy_type == "heuristic":
            # ---- your original policy (epsilon-greedy on memory scores) ----
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

        elif self.policy_type == "dqn":
            if self.dqn_agent is None:
                raise RuntimeError("policy_type='dqn' but no DQN agent was provided.")
            obs = self.build_obs(trial_index, step_in_trial, r_cur, theta_cur)
            action, self.dqn_hidden = self.dqn_agent.select_action(
                obs=obs,
                h=self.dqn_hidden,
                greedy=False,  # exploration controlled inside DQN agent
            )
            return int(action)

        elif self.policy_type == "gail":
            if self.gail is None:
                raise RuntimeError("policy_type='gail' but no GAIL object was provided.")
            obs = self.build_obs(trial_index, step_in_trial, r_cur, theta_cur)
            action, _logp, self.gail_hidden = self.gail.select_action(
                obs=obs,
                h=self.gail_hidden,
                greedy=False,
            )
            return int(action)

        else:
            raise ValueError(f"Unknown policy_type={self.policy_type!r}")

    def observe(self, action_port_idx: int, reward: int, trial_index: int):
        """
        Update internal agent state after action and reward.

        For now:
          - update last_visit_trials (T_r)
          - we do NOT modify prior_ports, to keep behaviour equal to notebook.

        DQN/GAIL training uses the transitions you log externally.
        """
        self.last_visit_trials[action_port_idx] = trial_index
        # reward could be used in future for learning belief / meta-parameters


# ---------------------------------------------------------------------- #
# Simulation over oversampled sessions                                   #
# ---------------------------------------------------------------------- #

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
    policy_type: Literal["heuristic", "dqn", "gail"] = "heuristic",
    dqn_agent: Optional[DQNRecurrentAgent] = None,
    gail: Optional[GAILRecurrent] = None,
) -> Dict[str, Any]:
    """
    Run the POMDP simulation over oversampled sessions.

    - If policy_type="heuristic": behaviour is identical to your current notebook.
    - If "dqn": actions are selected by the recurrent DQN agent.
    - If "gail": actions are selected by the recurrent GAIL policy.

    Returns
    -------
    dict with:
      - sim_pokes_sessions_pen      : list[sess][trials][ports (1..8)]
      - hist_per_trial_sessions_pen : list[sess] (T_s, 8) histograms
      - hist_per_session_pen        : list[sess] (8,) per-session histogram
      - tau_true_per_session_out    : list[int]
      - total_pokes                 : int
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
        policy_type=policy_type,
        dqn_agent=dqn_agent,
        gail=gail,
        max_trials=20,  # τ range
    )

    sim_pokes_sessions_pen: List[Any] = []
    hist_per_trial_sessions_pen: List[np.ndarray] = []
    hist_per_session_pen: List[np.ndarray] = []
    tau_list: List[int] = []
    total_pokes = 0

    for i in range(len(session_order)):
        current_distance = int(distance_seq_oversampled[i])
        current_arr_port = int(arr_ports_oversampled[i])   # correct port (1..8)
        original_session_index = int(session_order[i])
        current_positions_start = positionsStartTrial_polar_oversampled[i]

        T_max_ep = len(data_pokes_masked["all_pokes"][i])
        tau_true = int(tau_true_per_session[i])
        tau_list.append(tau_true)

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
        reward_flag = 0

        for t in range(T_max_ep):  # trial loop
            r_cur = float(start_polar[t, 0])
            theta_cur = float(start_polar[t, 1])

            trial_pokes: List[int] = []
            trial_hist = np.zeros(8, dtype=int)

            n_pokes_this_trial = len(data_pokes_masked["all_pokes"][i][t])
            n_pokes_this_trial = min(n_pokes_this_trial, max_pokes_per_trial)

            for s in range(n_pokes_this_trial):
                p = agent.select_action(
                    trial_index=t,
                    step_in_trial=s,
                    r_cur=r_cur,
                    theta_cur=theta_cur,
                )

                reward = 0
                if (p == current_arr_port - 1) and (t >= tau_true):
                    reward = 1
                    reward_flag = 1

                agent.observe(action_port_idx=p, reward=reward, trial_index=t)

                trial_pokes.append(p)
                trial_hist[p] += 1
                session_hist[p] += 1

                # after poke, animal is at that port
                r_cur = 1.0
                theta_cur = port_angles[p]

                if reward_flag == 1:
                    break

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

    # Convert 0..7 → 1..8 like your original sim
    sim_pokes_sessions_pen = [
        [[int(a) + 1 for a in trial] for trial in session]
        for session in sim_pokes_sessions_pen
    ]

    return {
        "sim_pokes_sessions_pen": sim_pokes_sessions_pen,
        "hist_per_trial_sessions_pen": hist_per_trial_sessions_pen,
        "hist_per_session_pen": hist_per_session_pen,
        "tau_true_per_session": tau_list,
        "total_pokes": total_pokes,
    }
