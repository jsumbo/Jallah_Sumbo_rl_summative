#!/usr/bin/env python3
"""
This script tests the environment functionality and creates a demonstration GIF.
"""

import sys
import os
import numpy as np
import imageio

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add environment path
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))

from custom_env import LiberianEntrepreneurshipEnv

def test_environment():
    """Test the basic functionality of the environment."""
    print("Testing Liberian Entrepreneurship Environment...")
    
    # Register and create environment
    env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")
    
    # Test random actions
    frames = []
    total_reward = 0
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Capture frame for GIF
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        print(f"Step {step}: Action={action}, Reward={reward:.2f}, Money=${info['money']:.1f}, Business Level={info['business_level']:.1f}")
        
        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break
    
    print(f"Total reward: {total_reward:.2f}")
    
    # Save demonstration GIF
    if frames:
        print("Saving demonstration GIF...")
        imageio.mimsave(os.path.join(PROJECT_ROOT, 'results', 'random_agent_demo.gif'), 
                        frames, duration=0.5)
        print("GIF saved successfully!")
    
    env.close()
    return True

def test_all_actions():
    """Test all possible actions in the environment."""
    print("\nTesting all actions...")
    
    env = LiberianEntrepreneurshipEnv()
    obs, info = env.reset()
    
    action_names = [
        "Move North", "Move Northeast", "Move East", "Move Southeast",
        "Move South", "Move Southwest", "Move West", "Move Northwest",
        "Study/Learn", "Apply for Loan", "Buy Inventory", "Sell Products",
        "Market Research", "Improve Customer Service"
    ]
    
    for action in range(env.action_space.n):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action {action} ({action_names[action]}): Reward={reward:.2f}")
        
        if terminated or truncated:
            obs, info = env.reset()
    
    env.close()

def demonstrate_strategic_play():
    """Demonstrate a more strategic approach to playing the game."""
    print("\nDemonstrating strategic gameplay...")
    
    env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    frames = []
    total_reward = 0
    
    # Strategic sequence: Go to school -> study -> go to supplier -> buy inventory -> go to market -> sell
    strategic_actions = [
        # Move to school (1,1)
        2, 2, 0, 0,  # Move to (1,1)
        8, 8, 8,     # Study multiple times
        # Move to supplier (3,7)
        1, 1, 2, 2, 2, 2,  # Move towards supplier
        10,          # Buy inventory
        # Move to market (2,3)
        3, 3, 3, 3,  # Move towards market
        11,          # Sell products
        12,          # Market research
        13           # Improve customer service
    ]
    
    for i, action in enumerate(strategic_actions):
        if i >= 30:  # Limit demonstration length
            break
            
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        print(f"Strategic Step {i}: Action={action}, Reward={reward:.2f}, Money=${info['money']:.1f}")
        
        if terminated or truncated:
            break
    
    print(f"Strategic play total reward: {total_reward:.2f}")
    
    # Save strategic demonstration GIF
    if frames:
        print("Saving strategic demonstration GIF...")
        imageio.mimsave(os.path.join(PROJECT_ROOT, 'results', 'strategic_demo.gif'), 
                       frames, duration=0.5)
        print("Strategic GIF saved successfully!")
    
    env.close()

if __name__ == "__main__":
    # Create results directory
    os.makedirs(os.path.join(PROJECT_ROOT, 'results'), exist_ok=True)
    
    # Run tests
    test_environment()
    test_all_actions()
    demonstrate_strategic_play()
    
    print("\nEnvironment testing completed successfully!")
    print("Check the results directory for demonstration GIFs.")

