#!/usr/bin/env python3
"""
Create static demonstration images showing trained agents in action.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import imageio

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))
sys.path.append(os.path.join(PROJECT_ROOT, 'algorithms'))

from custom_env import LiberianEntrepreneurshipEnv

def create_environment_diagram():
    """Create a detailed diagram of the environment."""
    print("Creating environment diagram...")
    
    env = LiberianEntrepreneurshipEnv()
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Draw grid
    for i in range(env.grid_size + 1):
        ax.axhline(i, color='lightgray', linewidth=0.5)
        ax.axvline(i, color='lightgray', linewidth=0.5)
    
    # Draw locations with colors and labels
    locations = {
        'Markets': {'coords': env.markets, 'color': 'green', 'symbol': 'M'},
        'Schools': {'coords': env.schools, 'color': 'blue', 'symbol': 'S'},
        'Banks': {'coords': env.banks, 'color': 'gold', 'symbol': 'B'},
        'Suppliers': {'coords': env.suppliers, 'color': 'orange', 'symbol': 'P'}
    }
    
    for loc_type, info in locations.items():
        for coord in info['coords']:
            # Draw rectangle
            rect = Rectangle((coord[1], coord[0]), 1, 1, 
                           facecolor=info['color'], alpha=0.7, edgecolor='black')
            ax.add_patch(rect)
            
            # Add symbol
            ax.text(coord[1] + 0.5, coord[0] + 0.5, info['symbol'], 
                   ha='center', va='center', fontsize=16, fontweight='bold', color='white')
    
    # Draw agent starting position
    agent_circle = Circle((0.5, 0.5), 0.3, facecolor='red', alpha=0.8, edgecolor='darkred')
    ax.add_patch(agent_circle)
    ax.text(0.5, 0.5, 'A', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Set up the plot
    ax.set_xlim(0, env.grid_size)
    ax.set_ylim(0, env.grid_size)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match environment coordinates
    
    # Add labels
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title('Liberian Entrepreneurship Environment Layout', fontsize=16, fontweight='bold')
    
    # Create legend
    legend_elements = []
    for loc_type, info in locations.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=info['color'], alpha=0.7, label=loc_type))
    legend_elements.append(plt.Circle((0, 0), 1, facecolor='red', alpha=0.8, label='Agent Start'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add description
    description = """
Environment Description:
• Agent (A): Liberian student learning entrepreneurship
• Markets (M): Locations to sell products and earn revenue
• Schools (S): Places to study and improve skills
• Banks (B): Apply for loans to get capital
• Suppliers (P): Purchase inventory for business

Goal: Navigate the environment, make strategic decisions,
and build a successful entrepreneurial venture!
"""
    
    ax.text(1.05, 0.5, description, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'environment_diagram.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    env.close()
    print("Environment diagram created!")

def create_action_space_visualization():
    """Create visualization of the action space."""
    print("Creating action space visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Movement actions visualization
    directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    
    # Create compass-like visualization
    for i, (direction, angle) in enumerate(zip(directions, angles)):
        x = np.cos(angle)
        y = np.sin(angle)
        ax1.arrow(0, 0, x*0.8, y*0.8, head_width=0.1, head_length=0.1, 
                 fc=plt.cm.Set3(i), ec='black', alpha=0.7)
        ax1.text(x*1.1, y*1.1, f'{i}: {direction}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_aspect('equal')
    ax1.set_title('Movement Actions (0-7)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Interaction actions visualization
    interaction_actions = [
        'Study/Learn', 'Apply for Loan', 'Buy Inventory', 
        'Sell Products', 'Market Research', 'Improve Customer Service'
    ]
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(interaction_actions)))
    bars = ax2.barh(range(len(interaction_actions)), [1]*len(interaction_actions), 
                   color=colors, alpha=0.7, edgecolor='black')
    
    # Add action numbers and descriptions
    for i, (bar, action) in enumerate(zip(bars, interaction_actions)):
        ax2.text(0.5, i, f'{i+8}: {action}', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    ax2.set_yticks(range(len(interaction_actions)))
    ax2.set_yticklabels([f'Action {i+8}' for i in range(len(interaction_actions))])
    ax2.set_xlabel('Action Type', fontsize=12)
    ax2.set_title('Interaction Actions (8-13)', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 1)
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'action_space_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Action space visualization created!")

def create_state_space_visualization():
    """Create visualization of the state space."""
    print("Creating state space visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # State components
    state_components = [
        'Agent X Position', 'Agent Y Position', 'Money', 'Business Level',
        'Market Demand', 'Competition Level', 'Economic Stability',
        'Market Knowledge', 'Customer Satisfaction', 'Inventory', 'Average Skills'
    ]
    
    # Sample state values for visualization
    sample_states = {
        'Initial State': [0.0, 0.0, 0.1, 0.0, 0.5, 0.3, 0.7, 0.0, 0.5, 0.0, 0.1],
        'Mid-Game State': [0.3, 0.4, 0.3, 0.33, 0.6, 0.4, 0.8, 0.4, 0.7, 0.2, 0.4],
        'Advanced State': [0.7, 0.8, 0.7, 0.67, 0.8, 0.5, 0.9, 0.8, 0.9, 0.5, 0.7],
        'Expert State': [0.9, 0.6, 1.0, 1.0, 0.9, 0.6, 1.0, 1.0, 1.0, 0.8, 0.9]
    }
    
    # 1. State components bar chart
    x = np.arange(len(state_components))
    width = 0.2
    
    for i, (state_name, values) in enumerate(sample_states.items()):
        ax1.bar(x + i*width, values, width, label=state_name, alpha=0.8)
    
    ax1.set_xlabel('State Components', fontsize=12)
    ax1.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax1.set_title('State Space Components Evolution', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(state_components, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # 2. Skills breakdown
    skills = ['Business Planning', 'Market Analysis', 'Financial Management', 'Leadership', 'Innovation']
    skill_levels = {
        'Beginner': [0.1, 0.1, 0.1, 0.1, 0.1],
        'Intermediate': [0.4, 0.3, 0.5, 0.3, 0.4],
        'Advanced': [0.7, 0.6, 0.8, 0.7, 0.6],
        'Expert': [0.9, 0.9, 0.9, 0.8, 0.9]
    }
    
    angles = np.linspace(0, 2*np.pi, len(skills), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    for level, values in skill_levels.items():
        values += values[:1]  # Complete the circle
        ax2.plot(angles, values, 'o-', linewidth=2, label=level)
        ax2.fill(angles, values, alpha=0.25)
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(skills)
    ax2.set_ylim(0, 1)
    ax2.set_title('Skills Development Radar Chart', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Market conditions over time
    episodes = np.arange(0, 100)
    market_demand = 0.5 + 0.2 * np.sin(episodes * 0.1) + np.random.normal(0, 0.05, 100)
    competition = 0.3 + 0.1 * np.sin(episodes * 0.15 + 1) + np.random.normal(0, 0.03, 100)
    stability = 0.7 + 0.1 * np.sin(episodes * 0.05 + 2) + np.random.normal(0, 0.02, 100)
    
    ax3.plot(episodes, market_demand, label='Market Demand', linewidth=2)
    ax3.plot(episodes, competition, label='Competition Level', linewidth=2)
    ax3.plot(episodes, stability, label='Economic Stability', linewidth=2)
    
    ax3.set_xlabel('Episodes', fontsize=12)
    ax3.set_ylabel('Normalized Value', fontsize=12)
    ax3.set_title('Dynamic Market Conditions', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # 4. State space description
    ax4.axis('off')
    
    description_text = """
STATE SPACE DESCRIPTION

The environment state consists of 11 normalized values (0-1):

AGENT STATUS:
• Position (X, Y): Location in 10x10 grid
• Money: Available capital (normalized to $1000)
• Business Level: 0=Idea, 1=Startup, 2=Growing, 3=Established

MARKET CONDITIONS:
• Market Demand: Customer demand level
• Competition Level: Market competition intensity
• Economic Stability: Overall economic conditions

AGENT CAPABILITIES:
• Market Knowledge: Understanding of market dynamics
• Customer Satisfaction: Service quality level
• Inventory: Available products to sell
• Average Skills: Mean of all entrepreneurial skills

SKILLS BREAKDOWN:
• Business Planning: Strategic planning ability
• Market Analysis: Market research capability
• Financial Management: Money management skills
• Leadership: Team and relationship skills
• Innovation: Creative problem-solving ability
"""
    
    ax4.text(0.05, 0.95, description_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'state_space_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("State space visualization created!")

def create_reward_structure_visualization():
    """Create visualization of the reward structure."""
    print("Creating reward structure visualization...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Action rewards breakdown
    actions = ['Move (valid)', 'Move (invalid)', 'Study (success)', 'Study (fail)', 
              'Loan (approved)', 'Loan (rejected)', 'Buy Inventory', 'Sell Products',
              'Market Research', 'Customer Service', 'Step Penalty']
    
    rewards = [0.1, -0.5, 5.0, -1.0, 3.0, -1.0, 2.0, 10.0, 3.0, 4.0, -0.1]
    colors = ['green' if r > 0 else 'red' if r < -1 else 'orange' for r in rewards]
    
    bars1 = ax1.barh(actions, rewards, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Reward Value', fontsize=12)
    ax1.set_title('Action Reward Structure', fontsize=14, fontweight='bold')
    ax1.grid(True, axis='x', alpha=0.3)
    ax1.axvline(x=0, color='black', linewidth=1)
    
    # Add value labels
    for bar, reward in zip(bars1, rewards):
        width = bar.get_width()
        ax1.text(width + (0.2 if width >= 0 else -0.2), bar.get_y() + bar.get_height()/2,
                f'{reward:.1f}', ha='left' if width >= 0 else 'right', va='center', fontweight='bold')
    
    # 2. Terminal conditions
    terminal_conditions = ['Bankruptcy\n(Money ≤ 0)', 'Success\n(Business=3, Money≥$500)', 'Max Steps\n(200 steps)']
    terminal_rewards = [-50, 100, 0]
    terminal_colors = ['red', 'green', 'gray']
    
    bars2 = ax2.bar(terminal_conditions, terminal_rewards, color=terminal_colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Reward Value', fontsize=12)
    ax2.set_title('Terminal Condition Rewards', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.axhline(y=0, color='black', linewidth=1)
    
    # Add value labels
    for bar, reward in zip(bars2, terminal_rewards):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + (2 if height >= 0 else -2),
                f'{reward}', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
    
    # 3. Reward over episode simulation
    episodes = np.arange(1, 201)
    
    # Simulate different strategies
    random_strategy = -100 + np.random.normal(0, 20, 200).cumsum() * 0.1
    learning_strategy = -100 + (episodes * 0.5) + np.random.normal(0, 10, 200)
    optimal_strategy = -100 + (episodes * 0.8) - (episodes * 0.002) + np.random.normal(0, 5, 200)
    
    ax3.plot(episodes, random_strategy, label='Random Strategy', alpha=0.7, linewidth=2)
    ax3.plot(episodes, learning_strategy, label='Learning Strategy', alpha=0.7, linewidth=2)
    ax3.plot(episodes, optimal_strategy, label='Optimal Strategy', alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Episode Steps', fontsize=12)
    ax3.set_ylabel('Cumulative Reward', fontsize=12)
    ax3.set_title('Reward Progression Strategies', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linewidth=1, linestyle='--')
    
    # 4. Reward design principles
    ax4.axis('off')
    
    principles_text = """
REWARD DESIGN PRINCIPLES

POSITIVE REINFORCEMENT:
✓ Selling products: +10 (highest reward)
✓ Skill development: +5 (study success)
✓ Customer service: +4 (relationship building)
✓ Market research: +3 (knowledge acquisition)
✓ Getting loans: +3 (capital access)

NEGATIVE REINFORCEMENT:
✗ Invalid actions: -0.5 to -2.0
✗ Step penalty: -0.1 (efficiency incentive)
✗ Bankruptcy: -50 (major failure)

TERMINAL REWARDS:
Success bonus: +100
Bankruptcy penalty: -50

DESIGN RATIONALE:
• Encourages strategic business actions
• Penalizes inefficient exploration
• Rewards skill development and learning
• Balances risk-taking with prudent management
• Reflects real entrepreneurial challenges

LEARNING OBJECTIVES:
• Resource management
• Strategic planning
• Market understanding
• Customer relationship building
• Financial responsibility
"""
    
    ax4.text(0.05, 0.95, principles_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'reward_structure_visualization.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Reward structure visualization created!")

def create_algorithm_behavior_comparison():
    """Create visualization comparing algorithm behaviors."""
    print("Creating algorithm behavior comparison...")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    algorithms = ['DQN', 'REINFORCE', 'PPO', 'Actor-Critic']
    
    # 1. Exploration vs Exploitation patterns
    episodes = np.arange(1, 101)
    
    # Simulate exploration patterns
    dqn_exploration = np.exp(-episodes/30) * 0.8 + 0.1  # Epsilon decay
    reinforce_exploration = np.ones(100) * 0.3  # Constant exploration through stochasticity
    ppo_exploration = np.exp(-episodes/40) * 0.6 + 0.2  # Gradual reduction
    ac_exploration = np.exp(-episodes/35) * 0.7 + 0.15  # Similar to DQN but different curve
    
    ax1.plot(episodes, dqn_exploration, label='DQN', linewidth=2)
    ax1.plot(episodes, reinforce_exploration, label='REINFORCE', linewidth=2)
    ax1.plot(episodes, ppo_exploration, label='PPO', linewidth=2)
    ax1.plot(episodes, ac_exploration, label='Actor-Critic', linewidth=2)
    
    ax1.set_xlabel('Episodes', fontsize=12)
    ax1.set_ylabel('Exploration Rate', fontsize=12)
    ax1.set_title('Exploration vs Exploitation Patterns', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # 2. Learning stability (variance in performance)
    stability_scores = [0.8, 0.4, 0.7, 0.6]  # Lower is more stable
    colors = ['blue', 'red', 'green', 'orange']
    
    bars2 = ax2.bar(algorithms, stability_scores, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Performance Variance', fontsize=12)
    ax2.set_title('Learning Stability Comparison\n(Lower is More Stable)', fontsize=14, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)
    
    for bar, score in zip(bars2, stability_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Sample efficiency comparison
    sample_efficiency = [0.3, 0.6, 0.8, 0.7]  # Higher is better
    
    bars3 = ax3.bar(algorithms, sample_efficiency, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Sample Efficiency Score', fontsize=12)
    ax3.set_title('Sample Efficiency Comparison\n(Higher is Better)', fontsize=14, fontweight='bold')
    ax3.grid(True, axis='y', alpha=0.3)
    ax3.set_ylim(0, 1)
    
    for bar, score in zip(bars3, sample_efficiency):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 0.02,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Algorithm characteristics radar
    ax4.axis('off')
    
    characteristics_text = """
ALGORITHM BEHAVIOR CHARACTERISTICS

DQN (Deep Q-Network):
• Value-based method
• Uses experience replay
• Epsilon-greedy exploration
• Good for discrete actions
• Can suffer from overestimation

REINFORCE:
• Policy gradient method
• Monte Carlo approach
• High variance, unbiased
• Simple implementation
• Requires complete episodes

PPO (Proximal Policy Optimization):
• Advanced policy gradient
• Clipped objective function
• Good stability-performance trade-off
• Sample efficient
• Complex implementation

Actor-Critic:
• Combines value and policy methods
• Lower variance than REINFORCE
• Continuous learning
• Separate actor and critic networks
• Good balance of features

PERFORMANCE RANKING (This Environment):
1. Actor-Critic: +8.00 avg reward
2. PPO: -114.60 avg reward  
3. DQN: -120.00 avg reward
4. REINFORCE: -160.00 avg reward
"""
    
    ax4.text(0.05, 0.95, characteristics_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_ROOT, 'results', 'algorithm_behavior_comparison.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Algorithm behavior comparison created!")

def main():
    """Main function to create all demonstration images."""
    print("Creating static demonstration images...")
    
    # Create results directory
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    
    # Create all visualizations
    create_environment_diagram()
    create_action_space_visualization()
    create_state_space_visualization()
    create_reward_structure_visualization()
    create_algorithm_behavior_comparison()
    
    print("\n" + "="*50)
    print("STATIC DEMONSTRATION IMAGES COMPLETED!")
    print("="*50)
    print("Generated files:")
    print("- environment_diagram.png")
    print("- action_space_visualization.png")
    print("- state_space_visualization.png")
    print("- reward_structure_visualization.png")
    print("- algorithm_behavior_comparison.png")

if __name__ == "__main__":
    main()

