import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        return x

class REINFORCEAgent:
    """REINFORCE (Monte Carlo Policy Gradient) Agent."""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network
        self.policy_network = PolicyNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
        # Training metrics
        self.episode_rewards = []
        self.policy_losses = []
        
        # Episode storage
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage."""
        self.log_probs = []
        self.rewards = []
    
    def act(self, state):
        """Choose action using policy network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_network(state)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store log probability for training
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update_policy(self):
        """Update policy using REINFORCE algorithm."""
        if not self.rewards:
            return
        
        # Calculate discounted returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensor and normalize
        returns = torch.FloatTensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Optimize
        self.optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Store loss
        self.policy_losses.append(policy_loss.item())
        
        # Reset episode storage
        self.reset_episode()
    
    def train(self, env, episodes=1000, max_steps=200):
        """Train the REINFORCE agent."""
        print("Training REINFORCE Agent...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            self.reset_episode()
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                self.store_reward(reward)
                state = next_state
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            # Update policy at the end of episode
            self.update_policy()
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        print("REINFORCE Training completed!")
        return self.episode_rewards
    
    def evaluate(self, env, episodes=10, render=False):
        """Evaluate the trained agent."""
        print("Evaluating REINFORCE Agent...")
        
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(200):
                # Use deterministic policy for evaluation
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs = self.policy_network(state_tensor)
                action = action_probs.argmax().item()
                
                state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                
                if render:
                    env.render()
                
                if terminated or truncated:
                    break
            
            total_rewards.append(episode_reward)
            print(f"Evaluation Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        print(f"REINFORCE Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return total_rewards, avg_reward, std_reward
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'policy_network_state_dict': self.policy_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'policy_losses': self.policy_losses
        }, filepath)
        print(f"REINFORCE model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.policy_losses = checkpoint['policy_losses']
        print(f"REINFORCE model loaded from {filepath}")
    
    def plot_training_results(self, save_path=None):
        """Plot training results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('REINFORCE Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot moving average
        if len(self.episode_rewards) > 100:
            moving_avg = [np.mean(self.episode_rewards[max(0, i-100):i+1]) 
                         for i in range(len(self.episode_rewards))]
            ax1.plot(moving_avg, 'r-', label='Moving Average (100 episodes)')
            ax1.legend()
        
        # Plot policy losses
        if self.policy_losses:
            ax2.plot(self.policy_losses)
            ax2.set_title('REINFORCE Policy Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Policy Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
        return fig 