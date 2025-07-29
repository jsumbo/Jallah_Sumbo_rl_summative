#!/usr/bin/env python3
"""
Training script for all RL algorithms.
This script trains DQN, REINFORCE, PPO, and Actor-Critic agents and compares their performance.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt 
import json
from datetime import datetime

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add environment path
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))
sys.path.append(os.path.join(PROJECT_ROOT, 'algorithms'))

from environment.custom_env import LiberianEntrepreneurshipEnv
from training.dqn_training import DQNAgent
from training.pg_training import REINFORCEAgent, PPOAgent, ActorCriticAgent

def create_environment():
    """Create and return the environment."""
    return LiberianEntrepreneurshipEnv()

def train_dqn(env, episodes=800):
    """Train DQN agent."""
    print("="*50)
    print("TRAINING DQN AGENT")
    print("="*50)
    
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        buffer_size=10000,
        batch_size=32,
        target_update=100
    )
    
    rewards = agent.train(env, episodes=episodes)
    
    # Save model
    agent.save_model(os.path.join(PROJECT_ROOT, 'results', 'dqn_model.pth'))
    
    # Plot results
    agent.plot_training_results(os.path.join(PROJECT_ROOT, 'results', 'dqn_training.png'))
    
    # Evaluate
    eval_rewards, avg_reward, std_reward = agent.evaluate(env, episodes=10)
    
    return agent, rewards, eval_rewards, avg_reward, std_reward

def train_reinforce(env, episodes=800):
    """Train REINFORCE agent."""
    print("="*50)
    print("TRAINING REINFORCE AGENT")
    print("="*50)
    
    agent = REINFORCEAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3,
        gamma=0.99
    )
    
    rewards = agent.train(env, episodes=episodes)
    
    # Save model
    agent.save_model(os.path.join(PROJECT_ROOT, 'results', 'reinforce_model.pth'))
    
    # Plot results
    agent.plot_training_results(os.path.join(PROJECT_ROOT, 'results', 'reinforce_training.png'))
    
    # Evaluate
    eval_rewards, avg_reward, std_reward = agent.evaluate(env, episodes=10)
    
    return agent, rewards, eval_rewards, avg_reward, std_reward

def train_ppo(env, episodes=800):
    """Train PPO agent."""
    print("="*50)
    print("TRAINING PPO AGENT")
    print("="*50)
    
    agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=4,
        c1=0.5,
        c2=0.01
    )
    
    rewards = agent.train(env, episodes=episodes)
    
    # Save model
    agent.save_model(os.path.join(PROJECT_ROOT, 'results', 'ppo_model.pth'))
    
    # Plot results
    agent.plot_training_results(os.path.join(PROJECT_ROOT, 'results', 'ppo_training.png'))
    
    # Evaluate
    eval_rewards, avg_reward, std_reward = agent.evaluate(env, episodes=10)
    
    return agent, rewards, eval_rewards, avg_reward, std_reward

def train_actor_critic(env, episodes=800):
    """Train Actor-Critic agent."""
    print("="*50)
    print("TRAINING ACTOR-CRITIC AGENT")
    print("="*50)
    
    agent = ActorCriticAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99
    )
    
    rewards = agent.train(env, episodes=episodes)
    
    # Save model
    agent.save_model(os.path.join(PROJECT_ROOT, 'results', 'actor_critic_model.pth'))
    
    # Plot results
    agent.plot_training_results(os.path.join(PROJECT_ROOT, 'results', 'actor_critic_training.png'))
    
    # Evaluate
    eval_rewards, avg_reward, std_reward = agent.evaluate(env, episodes=10)
    
    return agent, rewards, eval_rewards, avg_reward, std_reward

def compare_algorithms(results):
    """Compare the performance of all algorithms."""
    print("="*50)
    print("ALGORITHM COMPARISON")
    print("="*50)
    
    # Extract data
    algorithms = list(results.keys())
    training_rewards = {alg: results[alg]['training_rewards'] for alg in algorithms}
    eval_avg_rewards = {alg: results[alg]['eval_avg_reward'] for alg in algorithms}
    eval_std_rewards = {alg: results[alg]['eval_std_reward'] for alg in algorithms}
    
    # Print comparison table
    print(f"{'Algorithm':<15} {'Avg Eval Reward':<15} {'Std Eval Reward':<15} {'Final Training Reward':<20}")
    print("-" * 70)
    
    for alg in algorithms:
        final_training = np.mean(training_rewards[alg][-100:]) if len(training_rewards[alg]) >= 100 else np.mean(training_rewards[alg])
        print(f"{alg:<15} {eval_avg_rewards[alg]:<15.2f} {eval_std_rewards[alg]:<15.2f} {final_training:<20.2f}")
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    
    # Training rewards comparison
    for alg in algorithms:
        rewards = training_rewards[alg]
        # Calculate moving average
        window = min(100, len(rewards) // 10)
        if window > 1:
            moving_avg = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            ax1.plot(moving_avg, label=f'{alg} (Moving Avg)')
        else:
            ax1.plot(rewards, label=alg)
    
    ax1.set_title('Training Rewards Comparison')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Evaluation rewards comparison (bar plot)
    ax2.bar(algorithms, [eval_avg_rewards[alg] for alg in algorithms], 
            yerr=[eval_std_rewards[alg] for alg in algorithms], capsize=5)
    ax2.set_title('Evaluation Performance Comparison')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, axis='y')
    
    # Final training performance (last 100 episodes)
    final_performances = []
    for alg in algorithms:
        rewards = training_rewards[alg]
        final_perf = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
        final_performances.append(final_perf)
    
    ax3.bar(algorithms, final_performances)
    ax3.set_title('Final Training Performance (Last 100 Episodes)')
    ax3.set_ylabel('Average Reward')
    ax3.grid(True, axis='y')
    
    # Learning curves (first 200 episodes)
    for alg in algorithms:
        rewards = training_rewards[alg][:200]  # First 200 episodes
        ax4.plot(rewards, label=alg, alpha=0.7)
    
    ax4.set_title('Early Learning Curves (First 200 Episodes)')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'algorithm_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def save_results_summary(results):
    """Save results summary to JSON file."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'environment': 'LiberianEntrepreneurshipEnv',
        'algorithms': {}
    }
    
    for alg, data in results.items():
        summary['algorithms'][alg] = {
            'evaluation_avg_reward': float(data['eval_avg_reward']),
            'evaluation_std_reward': float(data['eval_std_reward']),
            'final_training_avg_reward': float(np.mean(data['training_rewards'][-100:]) 
                                             if len(data['training_rewards']) >= 100 
                                             else np.mean(data['training_rewards'])),
            'total_episodes': len(data['training_rewards']),
            'convergence_episode': None  # Could be calculated based on criteria
        }
    
    with open(os.path.join(PROJECT_ROOT, 'results', 'results_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Results summary saved to results_summary.json")

def main():
    """Main training function."""
    print("Starting comprehensive RL training on Liberian Entrepreneurship Environment")
    print("This may take a while...")
    
    # Create results directory
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    
    # Create environment
    env = create_environment()
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Training parameters
    episodes = 600  # Reduced for faster training
    
    results = {}
    
    # Train all algorithms
    try:
        # DQN
        dqn_agent, dqn_rewards, dqn_eval, dqn_avg, dqn_std = train_dqn(env, episodes)
        results['DQN'] = {
            'agent': dqn_agent,
            'training_rewards': dqn_rewards,
            'eval_rewards': dqn_eval,
            'eval_avg_reward': dqn_avg,
            'eval_std_reward': dqn_std
        }
        
        # REINFORCE
        reinforce_agent, reinforce_rewards, reinforce_eval, reinforce_avg, reinforce_std = train_reinforce(env, episodes)
        results['REINFORCE'] = {
            'agent': reinforce_agent,
            'training_rewards': reinforce_rewards,
            'eval_rewards': reinforce_eval,
            'eval_avg_reward': reinforce_avg,
            'eval_std_reward': reinforce_std
        }
        
        # PPO
        ppo_agent, ppo_rewards, ppo_eval, ppo_avg, ppo_std = train_ppo(env, episodes)
        results['PPO'] = {
            'agent': ppo_agent,
            'training_rewards': ppo_rewards,
            'eval_rewards': ppo_eval,
            'eval_avg_reward': ppo_avg,
            'eval_std_reward': ppo_std
        }
        
        # Actor-Critic
        ac_agent, ac_rewards, ac_eval, ac_avg, ac_std = train_actor_critic(env, episodes)
        results['Actor-Critic'] = {
            'agent': ac_agent,
            'training_rewards': ac_rewards,
            'eval_rewards': ac_eval,
            'eval_avg_reward': ac_avg,
            'eval_std_reward': ac_std
        }
        
        # Compare algorithms
        compare_algorithms(results)
        
        # Save results summary
        save_results_summary(results)
        
        print("="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("Check the results directory for:")
        print("- Individual model files (.pth)")
        print("- Training plots (.png)")
        print("- Algorithm comparison plot")
        print("- Results summary (JSON)")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()

if __name__ == "__main__":
    main()

