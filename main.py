import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'environment'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'training'))

from custom_env import LiberianEntrepreneurshipEnv
from dqn_training import DQNAgent
from pg_training import REINFORCEAgent, PPOAgent, ActorCriticAgent

def run_environment_test():
    print("Running environment test...")
    env = LiberianEntrepreneurshipEnv()
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        if terminated or truncated:
            break
    env.close()
    print("Environment test complete.")

def run_quick_train():
    print("Running quick training demo...")
    
    # Create environment
    env = LiberianEntrepreneurshipEnv()
    print(f"Environment created: {env.__class__.__name__}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Quick training parameters for demo
    episodes = 50  # Very short for quick demo
    max_steps = 100
    
    # Train DQN agent
    print(f"\nTraining DQN agent for {episodes} episodes...")
    dqn_agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.98,  # Faster decay for quick demo
        epsilon_min=0.1,
        buffer_size=1000,
        batch_size=16,
        target_update=25
    )
    
    # Train the agent
    training_rewards = dqn_agent.train(env, episodes=episodes, max_steps=max_steps)
    
    # Evaluate the trained agent
    print(f"\nEvaluating trained DQN agent...")
    eval_rewards, eval_avg, eval_std = dqn_agent.evaluate(env, episodes=5)
    
    # Print results
    print(f"\n{'='*40}")
    print("QUICK TRAINING RESULTS")
    print(f"{'='*40}")
    print(f"Training episodes: {episodes}")
    print(f"Final training reward (last 10 episodes): {np.mean(training_rewards[-10:]):.2f}")
    print(f"Evaluation average reward: {eval_avg:.2f} ± {eval_std:.2f}")
    print(f"Best evaluation episode: {max(eval_rewards):.2f}")
    
    # Create a simple plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(training_rewards, label='Training Rewards', alpha=0.7)
        plt.axhline(y=eval_avg, color='r', linestyle='--', label=f'Evaluation Avg: {eval_avg:.2f}')
        plt.fill_between(range(len(training_rewards)), 
                        [eval_avg - eval_std] * len(training_rewards),
                        [eval_avg + eval_std] * len(training_rewards),
                        alpha=0.2, color='r')
        plt.title('DQN Quick Training Demo')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        import os
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/quick_demo_training.png', dpi=300, bbox_inches='tight')
        print(f"Training plot saved to: results/quick_demo_training.png")
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot generation.")
    
    # Save the trained model
    try:
        os.makedirs('models/dqn', exist_ok=True)
        dqn_agent.save_model('models/dqn/demo_model.pth')
        print(f"Model saved to: models/dqn/demo_model.pth")
    except Exception as e:
        print(f"Could not save model: {e}")
    
    env.close()
    print(f"\nQuick training demo completed!")
    print(f"For full training with all algorithms, run: python quick_train.py")

def main():
    parser = argparse.ArgumentParser(description="Run RL experiments for Liberian Entrepreneurship Simulation.")
    parser.add_argument("--test_env", action="store_true", help="Run a quick test of the environment.")
    parser.add_argument("--quick_train", action="store_true", help="Run the quick training demo.")
    # Add more arguments for full training, evaluation, etc.

    args = parser.parse_args()

    if args.test_env:
        run_environment_test()
    elif args.quick_train:
        run_quick_train()
    else:
        print("Please specify an action: --test_env or --quick_train")
        print("For full training, run `python train_all_agents.py` directly.")

if __name__ == "__main__":
    main()


