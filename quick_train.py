#!/usr/bin/env python3
"""
Quick training script for demonstration purposes.
Trains all algorithms with reduced episodes for faster completion.
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

def main():
    """Quick training for demonstration."""
    print("Quick Training - Liberian Entrepreneurship Environment")
    
    # Create results directory
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    
    # Create environment
    env = LiberianEntrepreneurshipEnv()
    
    print(f"Environment: {env.__class__.__name__}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Quick training parameters
    episodes = 100  # Very reduced for demo
    
    results = {}
    
    # Train DQN
    print("\n" + "="*30)
    print("TRAINING DQN (Quick Demo)")
    print("="*30)
    
    dqn_agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.99,  # Faster decay for quick demo
        epsilon_min=0.1,
        buffer_size=1000,
        batch_size=16,
        target_update=50
    )
    
    dqn_rewards = dqn_agent.train(env, episodes=episodes)
    dqn_eval, dqn_avg, dqn_std = dqn_agent.evaluate(env, episodes=5)
    
    results['DQN'] = {
        'training_rewards': dqn_rewards,
        'eval_avg_reward': dqn_avg,
        'eval_std_reward': dqn_std
    }
    
    # Train REINFORCE
    print("\n" + "="*30)
    print("TRAINING REINFORCE (Quick Demo)")
    print("="*30)
    
    reinforce_agent = REINFORCEAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=1e-3,
        gamma=0.99
    )
    
    reinforce_rewards = reinforce_agent.train(env, episodes=episodes)
    reinforce_eval, reinforce_avg, reinforce_std = reinforce_agent.evaluate(env, episodes=5)
    
    results['REINFORCE'] = {
        'training_rewards': reinforce_rewards,
        'eval_avg_reward': reinforce_avg,
        'eval_std_reward': reinforce_std
    }
    
    # Train PPO
    print("\n" + "="*30)
    print("TRAINING PPO (Quick Demo)")
    print("="*30)
    
    ppo_agent = PPOAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        k_epochs=2,  # Reduced for speed
        c1=0.5,
        c2=0.01
    )
    
    ppo_rewards = ppo_agent.train(env, episodes=episodes)
    ppo_eval, ppo_avg, ppo_std = ppo_agent.evaluate(env, episodes=5)
    
    results['PPO'] = {
        'training_rewards': ppo_rewards,
        'eval_avg_reward': ppo_avg,
        'eval_std_reward': ppo_std
    }
    
    # Train Actor-Critic
    print("\n" + "="*30)
    print("TRAINING ACTOR-CRITIC (Quick Demo)")
    print("="*30)
    
    ac_agent = ActorCriticAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        lr_actor=1e-3,
        lr_critic=1e-3,
        gamma=0.99
    )
    
    ac_rewards = ac_agent.train(env, episodes=episodes)
    ac_eval, ac_avg, ac_std = ac_agent.evaluate(env, episodes=5)
    
    results['Actor-Critic'] = {
        'training_rewards': ac_rewards,
        'eval_avg_reward': ac_avg,
        'eval_std_reward': ac_std
    }
    
    # Create comparison plot
    print("\n" + "="*30)
    print("CREATING COMPARISON PLOTS")
    print("="*30)
    
    algorithms = list(results.keys())
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training rewards comparison
    for alg in algorithms:
        rewards = results[alg]['training_rewards']
        ax1.plot(rewards, label=alg, alpha=0.7)
    
    ax1.set_title('Training Rewards Comparison (Quick Demo)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.legend()
    ax1.grid(True)
    
    # Evaluation rewards comparison
    eval_rewards = [results[alg]['eval_avg_reward'] for alg in algorithms]
    eval_stds = [results[alg]['eval_std_reward'] for alg in algorithms]
    
    ax2.bar(algorithms, eval_rewards, yerr=eval_stds, capsize=5)
    ax2.set_title('Evaluation Performance Comparison')
    ax2.set_ylabel('Average Reward')
    ax2.grid(True, axis='y')
    
    # Final training performance
    final_performances = []
    for alg in algorithms:
        rewards = results[alg]['training_rewards']
        final_perf = np.mean(rewards[-20:]) if len(rewards) >= 20 else np.mean(rewards)
        final_performances.append(final_perf)
    
    ax3.bar(algorithms, final_performances)
    ax3.set_title('Final Training Performance (Last 20 Episodes)')
    ax3.set_ylabel('Average Reward')
    ax3.grid(True, axis='y')
    
    # Learning curves (smoothed)
    for alg in algorithms:
        rewards = results[alg]['training_rewards']
        # Simple moving average
        window = min(10, len(rewards) // 5)
        if window > 1:
            smoothed = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
            ax4.plot(smoothed, label=f'{alg} (smoothed)', linewidth=2)
        else:
            ax4.plot(rewards, label=alg)
    
    ax4.set_title('Smoothed Learning Curves')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Reward')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'quick_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print results summary
    print("\n" + "="*50)
    print("QUICK TRAINING RESULTS SUMMARY")
    print("="*50)
    print(f"{'Algorithm':<15} {'Eval Avg':<10} {'Eval Std':<10} {'Final Avg':<10}")
    print("-" * 50)
    
    for alg in algorithms:
        final_avg = np.mean(results[alg]['training_rewards'][-20:]) if len(results[alg]['training_rewards']) >= 20 else np.mean(results[alg]['training_rewards'])
        print(f"{alg:<15} {results[alg]['eval_avg_reward']:<10.2f} {results[alg]['eval_std_reward']:<10.2f} {final_avg:<10.2f}")
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'environment': 'LiberianEntrepreneurshipEnv',
        'training_episodes': episodes,
        'note': 'Quick demo training with reduced episodes',
        'algorithms': {}
    }
    
    for alg, data in results.items():
        summary['algorithms'][alg] = {
            'evaluation_avg_reward': float(data['eval_avg_reward']),
            'evaluation_std_reward': float(data['eval_std_reward']),
            'final_training_avg_reward': float(np.mean(data['training_rewards'][-20:]) 
                                             if len(data['training_rewards']) >= 20 
                                             else np.mean(data['training_rewards'])),
            'total_episodes': len(data['training_rewards'])
        }
    
    with open(os.path.join(PROJECT_ROOT, 'results', 'quick_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nQuick training completed successfully!")
    print(f"Results saved to: {os.path.join(PROJECT_ROOT, 'results')}")
    
    env.close()

if __name__ == "__main__":
    main()

