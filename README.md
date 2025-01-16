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
  - Selected actions.

## How to Use the Model

1. Clone the Repository

   ```bash
   git clone https://<GITHUB TOKEN>:x-oauth-basic@github.com/Jercog-team/MemoryRL
   cd MouseBehaviorPOMDP

2. Install Dependencies

    Ensure you have Python installed along with the required libraries:

    ```bash
    pip install numpy matplotlib


3. Run the Simulation
   
   ```bash
    python mouse_pomdp.py

This will execute the simulation and generate the visualizations.

