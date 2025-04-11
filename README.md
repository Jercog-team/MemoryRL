# MemoryRL


A Python implementation of a Partially Observable Markov Decision Process (POMDP) to simulate mouse behavior in a multi-port experiment. The goal is to mimic how a mouse learns to find the correct port where water becomes available after a delay, while also displaying exploratory behavior.

## Features
- Simulates a mouse's decision-making process to find water in one of several ports.
- Implements Bayesian belief updates and Q-value-based action selection.
- Includes plotting functions to visualize:
  - Cumulative rewards.
  - Actions taken by the mouse.
  - Belief state evolution.
  - Observations and distance to the target port.


## Theoretical Background

### Partially Observable Markov Decision Process (POMDP)

  A POMDP is a framework used for decision-making when the agent has limited observability of the environment. It consists of:

  - States (S): The true underlying condition of the environment (in this case, which port is correct).

  - Actions (A): The choices available to the agent (which port to poke).

  - Observations (O): Noisy, indirect information about the state (distance to the correct port, availability of water).

  - Transition Model (T): The probability of moving from one state to another.

  - Observation Model (O): The probability of receiving a particular observation given a state and action.

  - Reward Function (R): The numerical feedback given for different actions.


### 8-Port Maze Task

This model is designed to replicate the 8-Port Maze Task, a standard experimental setup in behavioral neuroscience used to study spatial learning and decision-making in rodents. 

![8Port Maze Experiment](img/8port.png)

The key elements of this task include:

  - A circular arena with 8 equally spaced ports.
    
  - Each day is split into two phases: training and recall.

      - Training phase: The mouse learns the correct port by interacting with the environment.
      
      - Recall phase: The mouse attempts to remember the correct port based on prior experience.
  
  - The mouse starts from a central location and can poke different ports.
  
  - Only one port provides water, and it becomes available when the animal come into the trigger zone.
  
  - Rodents exhibit exploratory behavior, often poking nearby ports before consistently selecting the correct one.
  
  - Learning is assessed by tracking how quickly the subject shifts from exploration to consistent selection of the correct port.

This model aims to capture these dynamics by using a POMDP framework, where the agent must infer the correct port over time based on noisy observations and rewards.

### Behavioral Modeling

  In neuroscience and psychology, reinforcement learning (RL) is used to model how animals learn from rewards and adapt their behavior. This model follows trial-and-error learning, where the mouse gradually increases its preference for the correct port while still exhibiting exploration.

## Model Implementation

### States  
  The hidden state at each episode is a tuple:

$$
s = (r, \tau)
$$

  Where:
  - $\( r \in \{0, \dots, 7\} \)$ is the reward port.
  - $\( \tau \in \{0, \dots, T-1\} \)$ is the trial in which the reward becomes available.
  - The total number of hidden states is $\( 8 \times T \)$.
  - The state is static during an episode; the transition matrix is the identity.

### Actions
  At each poke within a trial, the agent selects a port to poke:

$$
a_t \in \{0, \dots, 7\}
$$

### Observations
  The observation after each poke is binary:

$$
o_t \in \{0, 1\}
$$

  Reward is received only if the poke hits the correct port *after* reward becomes available:

$$
P(o_t = 1 \mid a_t, r, \tau) = \mathbb{1}(a_t = r) \cdot \mathbb{1}(t \geq \tau)
$$

### Beliefs  
  The agent maintains a belief distribution over the hidden state:

$$
b_t(r, \tau) = P(r, \tau \mid o_{1:t}, a_{1:t})
$$

  The marginal belief over ports is:

$$
m_t(r) = \sum_{\tau} b_t(r, \tau)
$$

### Reward Function
  The agent receives a reward of 1 if water is delivered:

$$
r_t = o_t
$$

  The episode terminates immediately after the first reward is received.

---

### Policy Options

- **Marginal Belief (Greedy)**  
  Chooses the port with the highest marginal belief:

$$
a_t = \arg\max_r m_t(r)
$$

- **Last Visit**  
  Uses a mixture of the prior and the recency of visits to select the next action.

- **Bellman Optimal**  
  Computes the action that maximizes the expected future return based on recursive Q-values.

  - **Termination probability**:

  <div align="center">$$T_t(p) = m_t(p) \cdot \frac{t - \text{last\_visit}(p)}{T - \text{last\_visit}(p)}$$</div>

  - **Q-value recursion**:

$$
Q(s, a) = T_t(p) + \gamma \cdot (1 - T_t(p)) \cdot \max_{a'} Q(s', a')
$$

---

### Transition Model

- The hidden state $\( s = (r, \tau) \)$ is static.
- Belief transitions are deterministic given the action and outcome.
- If a reward is received, the agent enters a terminal state and the episode ends.

---

### Termination

The episode ends immediately upon receiving the first water reward:

$$
o_t = 1 \Rightarrow \text{terminate}
$$


## Results and Visualizations

### Pokes histogram


![Pokes histogram](img/hist.png)

### Pokes Histogram by distance between yesterday's and today's ports


![Pokes Histogram by distance](img/hist_dist.png)

### Transition matrix between ports at the same trial


![Transition matrix](img/trans_matrix.png)



## Updated Project Structure

The project is now organized as follows:

```
MemoryRL/
├── main.py                
├── README.md              
├── setup.py               
├── memoryrl/              
│   ├── __init__.py        
│   ├── agents/            
│   │   ├── __init__.py    
│   │   └── agents.py      
│   ├── pomdp/             
│   │   ├── __init__.py    
│   │   ├── belief.py      
│   │   └── policy.py      
│   ├── simulations/       
│   │   ├── __init__.py    
│   │   └── simulations.py 
│   ├── utils/             
│   │   ├── __init__.py    
│   │   ├── data_processing.py 
│   │   ├── filtering_functions.py 
│   │   ├── HMM_EM.py      
│   │   └── memoryIndex_functions.py 
│   └── visualization/     
│       ├── __init__.py    
│       ├── animation.py   
│       └── plotting.py    
├── img/                   
│   ├── 8port.png          
│   ├── hist.png           
│   ├── hist_dist.png           
│   ├── trans_matrix.png
└── __pycache__/           
```

## Updated Instructions

1. Clone the Repository

   ```bash
   git clone https://<GITHUB TOKEN>:x-oauth-basic@github.com/Jercog-team/MemoryRL
   cd MemoryRL
   ```

2. Install Dependencies

   Ensure you have Python installed along with the required libraries:

   ```bash
   pip install -e .
   ```

3. Run the Simulation

   ```bash
   python main.py
   ```

This will execute the simulation and generate the visualizations.

