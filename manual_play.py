#!/usr/bin/env python3
"""
Simple Manual Play Script for Liberian Entrepreneurship Environment
Allows you to control the agent step-by-step with clear instructions.
"""

import sys
import os
import numpy as np

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Add paths
sys.path.append(os.path.join(PROJECT_ROOT, 'environment'))

from environment.custom_env import LiberianEntrepreneurshipEnv

def print_instructions():
    """Print game instructions."""
    print("\n" + "="*60)
    print("🎮 LIBERIAN ENTREPRENEURSHIP - MANUAL PLAY")
    print("="*60)
    print("🎯 GOAL: Build a successful business!")
    print("   • Reach Business Level 3 with at least $500")
    print("   • Avoid bankruptcy (money ≤ 0)")
    print("   • Complete within 200 steps")
    
    print("\n📍 LOCATIONS:")
    print("   • 🟢 Markets (Green): Shopkeepers - Sell products for money")
    print("   • 🔵 Schools (Blue): Teachers - Study to improve skills")
    print("   • 🟡 Banks (Yellow): Bankers - Apply for loans")
    print("   • 🟠 Suppliers (Orange): Merchants - Buy inventory")
    
    print("\n🎮 CONTROLS:")
    print("   Movement (0-7):")
    print("   0=North, 1=NE, 2=East, 3=SE, 4=South, 5=SW, 6=West, 7=NW")
    print("   Actions (8-13):")
    print("   8=Study, 9=Loan, 10=Buy, 11=Sell, 12=Research, 13=Service")
    print("   Commands:")
    print("   'q' = Quit, 'h' = Help, 's' = Show status")
    print("="*60)

def print_status(env, info, step, total_reward):
    """Print current game status."""
    print(f"\n📊 STEP {step} - STATUS:")
    print(f"   💰 Money: ${info['money']:.1f}")
    print(f"   🏢 Business Level: {info['business_level']:.1f}")
    print(f"   📦 Inventory: {info.get('inventory', 0)}")
    print(f"   🎯 Total Reward: {total_reward:.1f}")
    print(f"   📍 Position: ({env.agent_pos[0]}, {env.agent_pos[1]})")
    
    # Show skills
    skills = info['skills']
    print(f"   🧠 Skills: Business({skills['business_planning']:.2f}) "
          f"Market({skills['market_analysis']:.2f}) "
          f"Finance({skills['financial_management']:.2f}) "
          f"Leadership({skills['leadership']:.2f}) "
          f"Innovation({skills['innovation']:.2f})")
    
    # Show nearby locations
    print(f"   📍 Nearby:")
    for i, market in enumerate(env.markets):
        if tuple(env.agent_pos) == market:
            print(f"      🟢 Market {i+1} (HERE) - Shopkeeper")
    for i, school in enumerate(env.schools):
        if tuple(env.agent_pos) == school:
            print(f"      🔵 School {i+1} (HERE) - Teacher")
    for i, bank in enumerate(env.banks):
        if tuple(env.agent_pos) == bank:
            print(f"      🟡 Bank {i+1} (HERE) - Banker")
    for i, supplier in enumerate(env.suppliers):
        if tuple(env.agent_pos) == supplier:
            print(f"      🟠 Supplier {i+1} (HERE) - Merchant")

def print_action_help():
    """Print action help."""
    print("\n🎯 ACTION GUIDE:")
    print("   Movement (0-7): Move to different locations")
    print("   8 (Study): Improve skills (costs $10, must be at school with teacher)")
    print("   9 (Loan): Apply for loan (must be at bank with banker)")
    print("   10 (Buy): Purchase inventory (must be at supplier with merchant)")
    print("   11 (Sell): Sell products (must be at market with shopkeeper)")
    print("   12 (Research): Market research (costs $5)")
    print("   13 (Service): Improve customer service (costs $8)")

def main():
    """Main manual play function."""
    print_instructions()
    
    # Initialize environment
    env = LiberianEntrepreneurshipEnv()
    obs, info = env.reset()
    
    total_reward = 0
    step = 0
    
    print(f"\n🚀 Starting with ${info['money']:.1f} and Business Level {info['business_level']:.1f}")
    print("👤 You are a young Liberian student learning entrepreneurship!")
    
    while True:
        print_status(env, info, step, total_reward)
        
        # Get user input
        action_input = input(f"\n🎮 Step {step + 1} - Choose action (0-13, 'h' for help, 'q' to quit): ").strip().lower()
        
        if action_input == 'q':
            print("👋 Thanks for playing!")
            break
        elif action_input == 'h':
            print_action_help()
            continue
        elif action_input == 's':
            print_status(env, info, step, total_reward)
            continue
        
        try:
            action = int(action_input)
            if action < 0 or action > 13:
                print("❌ Invalid action! Please choose 0-13.")
                continue
        except ValueError:
            print("❌ Invalid input! Please enter a number 0-13, 'h' for help, or 'q' to quit.")
            continue
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        # Print action result
        action_names = [
            "North", "NE", "East", "SE", "South", "SW", "West", "NW",
            "Study", "Loan", "Buy", "Sell", "Research", "Service"
        ]
        print(f"✅ Action: {action_names[action]} | Reward: {reward:.1f}")
        
        # Check game end
        if terminated:
            print(f"\n🎯 GAME ENDED!")
            print(f"   Final Money: ${info['money']:.1f}")
            print(f"   Final Business Level: {info['business_level']:.1f}")
            print(f"   Total Reward: {total_reward:.1f}")
            
            if info['money'] <= 0:
                print("❌ BANKRUPTCY! You ran out of money!")
            elif info['business_level'] >= 3 and info['money'] >= 500:
                print("🎉 SUCCESS! You built a successful business!")
            else:
                print("⏰ Time's up! You didn't reach the goal.")
            break
        elif truncated:
            print(f"\n⏰ Episode truncated at step {step}")
            break
    
    env.close()

if __name__ == "__main__":
    main() 