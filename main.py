from src.mouse_behavior_pomdp import MouseBehaviorPOMDP

def main():
    print('*********************************************** DAY 1 ***********************************************')
    agent_type = input("Choose agent type (1 or 2): ")
    if agent_type == "1":
        pomdp = MouseBehaviorPOMDP(num_ports=8, observation_noise=0.1, discount_factor=0.95, seed=42)
    else:
        pomdp = MouseBehaviorPOMDP(num_ports=8, observation_noise=0.2, discount_factor=0.90, seed=42)  # Example variation
    
    rewards, actions, observations, belief_history = pomdp.simulate(num_steps=50)
    pomdp.plot_behavior(rewards, actions, observations, belief_history)

    print('*********************************************** DAY 2 ***********************************************')
    pomdp.reset(keep_belief=True)
    rewards, actions, observations, belief_history = pomdp.simulate(num_steps=50)
    pomdp.plot_behavior(rewards, actions, observations, belief_history)

if __name__ == "__main__":
    main()
