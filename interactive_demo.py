#!/usr/bin/env python3
"""
Interactive Demo Script for Liberian Entrepreneurship Environment
Allows users to play with the environment, watch trained agents, and explore features.
"""

import sys
import os
import numpy as np
import pygame
import time

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))
sys.path.append(os.path.join(PROJECT_ROOT, 'training'))

from environment.custom_env import LiberianEntrepreneurshipEnv
from training.dqn_training import DQNAgent
from training.pg_training import REINFORCEAgent, PPOAgent, ActorCriticAgent

def print_menu():
    """Print the interactive demo menu."""
    print("\n" + "="*50)
    print("LIBERIAN ENTREPRENEURSHIP - INTERACTIVE DEMO")
    print("="*50)
    print("1. Watch Random Agent Play")
    print("2. Watch Strategic Agent Play")
    print("3. Watch Trained DQN Agent")
    print("4. Watch Trained REINFORCE Agent")
    print("5. Watch Trained PPO Agent")
    print("6. Watch Trained Actor-Critic Agent")
    print("7. Manual Play Mode (You Control the Agent)")
    print("8. Environment Information")
    print("9. Exit")
    print("="*50)

def watch_agent(env, agent_type="random", model_path=None, episodes=1):
    """Watch an agent play the environment."""
    print(f"\nWatching {agent_type.upper()} agent play...")
    print("Press any key to continue...")
    
    # Load trained agent if specified
    agent = None
    if model_path and os.path.exists(model_path):
        if "dqn" in model_path:
            agent = DQNAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n
            )
            agent.load_model(model_path)
        elif "pg" in model_path:
            agent = REINFORCEAgent(
                state_size=env.observation_space.shape[0],
                action_size=env.action_space.n
            )
            agent.load_model(model_path)
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Starting money: ${info['money']:.1f}")
        print(f"Starting business level: {info['business_level']:.1f}")
        
        while True:
            # Choose action
            if agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "strategic":
                action = choose_strategic_action(obs, info)
            else:
                # Use trained agent
                action = agent.act(obs, training=False)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print step info
            action_names = [
                "North", "NE", "East", "SE", "South", "SW", "West", "NW",
                "Study", "Loan", "Buy", "Sell", "Research", "Service"
            ]
            print(f"Step {step}: {action_names[action]} | Reward: {reward:.1f} | Money: ${info['money']:.1f} | Business: {info['business_level']:.1f}")
            
            if terminated or truncated:
                break
        
        print(f"Episode ended! Total reward: {total_reward:.1f}")
        print(f"Final money: ${info['money']:.1f}")
        print(f"Final business level: {info['business_level']:.1f}")
        
        if terminated:
            if info['money'] <= 0:
                print("❌ BANKRUPTCY - Agent ran out of money!")
            elif info['business_level'] >= 3 and info['money'] >= 500:
                print("✅ SUCCESS - Agent built a successful business!")
            else:
                print("⏰ TIME UP - Episode reached maximum steps")
        else:
            print("⏰ TIME UP - Episode truncated")
    
    input("\nPress Enter to continue...")

def choose_strategic_action(obs, info):
    """Choose a strategic action based on current state."""
    # Simple strategic logic
    money = info['money']
    business_level = info['business_level']
    skills = info['skills']
    avg_skills = np.mean(list(skills.values()))
    
    # If low on money and skills, study
    if money < 20 and avg_skills < 0.3:
        return 8  # Study
    
    # If low on money, try to get loan
    if money < 15:
        return 9  # Apply for loan
    
    # If have money and low inventory, buy inventory
    if money > 20 and obs[9] < 0.3:  # inventory
        return 10  # Buy inventory
    
    # If have inventory, sell products
    if obs[9] > 0.2:  # inventory
        return 11  # Sell products
    
    # If business is growing, do market research
    if business_level > 1:
        return 12  # Market research
    
    # If customer satisfaction is low, improve service
    if obs[8] < 0.6:  # customer satisfaction
        return 13  # Improve customer service
    
    # Default: move randomly
    return np.random.randint(0, 8)

def manual_play():
    """Allow user to manually control the agent."""
    print("\n" + "="*50)
    print("MANUAL PLAY MODE")
    print("="*50)
    print("You control the agent! Use the following keys:")
    print("Movement: W(0) A(3) S(4) D(1) - or arrow keys")
    print("Diagonal: Q(7) E(2) Z(6) C(5)")
    print("Actions: 1-Study 2-Loan 3-Buy 4-Sell 5-Research 6-Service")
    print("Press 'q' to quit")
    print("="*50)
    
    env = LiberianEntrepreneurshipEnv(render_mode="human")
    obs, info = env.reset()
    
    total_reward = 0
    step = 0
    
    print(f"Starting money: ${info['money']:.1f}")
    print(f"Starting business level: {info['business_level']:.1f}")
    
    while True:
        # Get user input
        action_input = input(f"\nStep {step + 1} - Choose action (0-13, or 'q' to quit): ").strip().lower()
        
        if action_input == 'q':
            break
        
        try:
            action = int(action_input)
            if action < 0 or action > 13:
                print("Invalid action! Please choose 0-13.")
                continue
        except ValueError:
            print("Invalid input! Please enter a number 0-13 or 'q'.")
            continue
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Print results
        action_names = [
            "North", "NE", "East", "SE", "South", "SW", "West", "NW",
            "Study", "Loan", "Buy", "Sell", "Research", "Service"
        ]
        print(f"Action: {action_names[action]} | Reward: {reward:.1f} | Money: ${info['money']:.1f} | Business: {info['business_level']:.1f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended! Total reward: {total_reward:.1f}")
            if terminated:
                if info['money'] <= 0:
                    print("❌ BANKRUPTCY - You ran out of money!")
                elif info['business_level'] >= 3 and info['money'] >= 500:
                    print("✅ SUCCESS - You built a successful business!")
            break
    
    env.close()

def show_environment_info():
    """Show detailed information about the environment."""
    print("\n" + "="*50)
    print("ENVIRONMENT INFORMATION")
    print("="*50)
    
    env = LiberianEntrepreneurshipEnv()
    
    print("🎯 GOAL: Build a successful business by reaching Business Level 3 with at least $500")
    print("\n📍 LOCATIONS:")
    print("• Markets (Green): Sell products and earn money")
    print("• Schools (Blue): Study to improve skills")
    print("• Banks (Yellow): Apply for loans to get capital")
    print("• Suppliers (Orange): Buy inventory for your business")
    
    print("\n🎮 ACTIONS:")
    print("Movement (0-7): North, NE, East, SE, South, SW, West, NW")
    print("Study (8): Improve entrepreneurial skills")
    print("Loan (9): Apply for business loan")
    print("Buy (10): Purchase inventory from suppliers")
    print("Sell (11): Sell products at markets")
    print("Research (12): Conduct market research")
    print("Service (13): Improve customer service")
    
    print("\n💰 REWARDS:")
    print("• Successful actions: +2 to +10 points")
    print("• Failed actions: -1 to -2 points")
    print("• Movement: +0.1 (valid), -0.5 (invalid)")
    print("• Success bonus: +100 points")
    print("• Bankruptcy penalty: -50 points")
    
    print("\n📊 STATE SPACE (11 values):")
    print("• Agent position (X, Y)")
    print("• Money (normalized)")
    print("• Business level (0-3)")
    print("• Market conditions (demand, competition, stability)")
    print("• Agent capabilities (knowledge, satisfaction, inventory, skills)")
    
    print("\n🎯 SUCCESS CONDITIONS:")
    print("• Business Level ≥ 3 AND Money ≥ $500")
    print("• OR reach maximum steps (200)")
    print("• OR run out of money (bankruptcy)")
    
    env.close()

def record_video_demo(agent_type, episodes, duration):
    """Record a video demonstration of agent behavior."""
    print(f"Recording {episodes} episodes of {agent_type} agent for {duration} seconds...")
    
    # Set up video recording
    import imageio
    import os
    
    # Create output directory
    os.makedirs("results", exist_ok=True)
    output_path = f"results/{agent_type}_demo.mp4"
    
    # Initialize environment and agent
    env = LiberianEntrepreneurshipEnv(render_mode="rgb_array")
    
    # Load appropriate agent
    agent = None
    if agent_type == "dqn":
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
        if os.path.exists("models/dqn/dqn_model.pth"):
            agent.load_model("models/dqn/dqn_model.pth")
    elif agent_type == "reinforce":
        agent = REINFORCEAgent(env.observation_space.shape[0], env.action_space.n)
        if os.path.exists("models/pg/reinforce_model.pth"):
            agent.load_model("models/pg/reinforce_model.pth")
    elif agent_type == "ppo":
        from training.ppo_agent import PPOAgent
        agent = PPOAgent(env.observation_space.shape[0], env.action_space.n)
        if os.path.exists("models/pg/ppo_model.pth"):
            agent.load_model("models/pg/ppo_model.pth")
    elif agent_type == "actor_critic":
        from training.actor_critic_agent import ActorCriticAgent
        agent = ActorCriticAgent(env.observation_space.shape[0], env.action_space.n)
        if os.path.exists("models/pg/actor_critic_model.pth"):
            agent.load_model("models/pg/actor_critic_model.pth")
    
    frames = []
    start_time = time.time()
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        step = 0
        
        print(f"Recording episode {episode + 1}/{episodes}")
        
        while time.time() - start_time < duration and step < 200:
            # Choose action
            if agent_type == "random":
                action = env.action_space.sample()
            elif agent_type == "strategic":
                action = choose_strategic_action(obs, info)
            else:
                action = agent.act(obs, training=False)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Capture frame
            frame = env.render()
            frames.append(frame)
            
            if terminated or truncated:
                break
    
    # Save video
    if frames:
        imageio.mimsave(output_path, frames, fps=4)
        print(f"Video saved to: {output_path}")
        print(f"Total frames: {len(frames)}")
        print(f"Duration: {len(frames)/4:.1f} seconds")
    else:
        print("No frames captured!")
    
    env.close()

def main():
    """Main interactive demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Liberian Entrepreneurship Interactive Demo")
    parser.add_argument("--record_video", action="store_true", help="Record video of agent demonstrations")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    parser.add_argument("--duration", type=int, default=180, help="Duration of video in seconds")
    parser.add_argument("--agent_type", type=str, default="actor_critic", 
                       choices=["random", "strategic", "dqn", "reinforce", "ppo", "actor_critic"],
                       help="Type of agent to demonstrate")
    
    args = parser.parse_args()
    
    if args.record_video:
        record_video_demo(args.agent_type, args.episodes, args.duration)
        return
    
    print("Welcome to the Liberian Entrepreneurship Interactive Demo!")
    print("This demo allows you to explore the environment and watch different agents play.")
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (1-9): ").strip()
            
            if choice == "1":
                env = LiberianEntrepreneurshipEnv(render_mode="human")
                watch_agent(env, "random")
                env.close()
                
            elif choice == "2":
                env = LiberianEntrepreneurshipEnv(render_mode="human")
                watch_agent(env, "strategic")
                env.close()
                
            elif choice == "3":
                env = LiberianEntrepreneurshipEnv(render_mode="human")
                model_path = "models/dqn/demo_model.pth"
                if os.path.exists(model_path):
                    watch_agent(env, "trained", model_path)
                else:
                    print("❌ No trained DQN model found! Run training first.")
                env.close()
                
            elif choice == "4":
                env = LiberianEntrepreneurshipEnv(render_mode="human")
                model_path = "models/pg/pg_model.pth"
                if os.path.exists(model_path):
                    watch_agent(env, "trained", model_path)
                else:
                    print("❌ No trained REINFORCE model found! Run training first.")
                env.close()
                
            elif choice == "5":
                print("⚠️ PPO agent demo not implemented yet. Use REINFORCE instead.")
                
            elif choice == "6":
                print("⚠️ Actor-Critic agent demo not implemented yet. Use REINFORCE instead.")
                
            elif choice == "7":
                manual_play()
                
            elif choice == "8":
                show_environment_info()
                
            elif choice == "9":
                print("Thanks for playing! Goodbye!")
                break
                
            else:
                print("Invalid choice! Please enter 1-9.")
                
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main() 