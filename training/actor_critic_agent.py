import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class ActorNetwork(nn.Module):
    """Actor network for Actor-Critic algorithm."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorNetwork, self).__init__()
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

class CriticNetwork(nn.Module):
    """Critic network for Actor-Critic algorithm."""
    
    def __init__(self, state_size, hidden_size=128):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ActorCriticAgent:
    """Actor-Critic Agent with separate networks."""
    
    def __init__(self, state_size, action_size, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.actor = ActorNetwork(state_size, action_size).to(self.device)
        self.critic = CriticNetwork(state_size).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Training metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # Episode storage
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage."""
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.entropies = []
    
    def act(self, state):
        """Choose action using actor network."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action probabilities and state value
        action_probs = self.actor(state)
        state_value = self.critic(state)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        
        # Store for training
        self.log_probs.append(dist.log_prob(action))
        self.values.append(state_value)
        self.entropies.append(dist.entropy())
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward for current step."""
        self.rewards.append(reward)
    
    def update_networks(self):
        """Update actor and critic networks."""
        if not self.rewards:
            return
        
        # Calculate returns
        returns = []
        G = 0
        for reward in reversed(self.rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # Convert to tensors
        returns = torch.FloatTensor(returns).to(self.device)
        log_probs = torch.stack(self.log_probs)
        values = torch.stack(self.values).squeeze()
        entropies = torch.stack(self.entropies)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages
        advantages = returns - values.detach()
        
        # Actor loss (policy gradient with baseline)
        actor_loss = -(log_probs * advantages.detach()).mean() - 0.01 * entropies.mean()
        
        # Critic loss (value function approximation)
        critic_loss = F.mse_loss(values, returns)
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        
        # Reset episode storage
        self.reset_episode()
    
    def train(self, env, episodes=1000, max_steps=200):
        """Train the Actor-Critic agent."""
        print("Training Actor-Critic Agent...")
        
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
            
            # Update networks at the end of episode
            self.update_networks()
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        print("Actor-Critic Training completed!")
        return self.episode_rewards
    
    def evaluate(self, env, episodes=10, render=False):
        """Evaluate the trained agent."""
        print("Evaluating Actor-Critic Agent...")
        
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(200):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs = self.actor(state_tensor)
                
                # Use deterministic policy for evaluation
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
        
        print(f"Actor-Critic Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return total_rewards, avg_reward, std_reward
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses
        }, filepath)
        print(f"Actor-Critic model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        print(f"Actor-Critic model loaded from {filepath}")
    
    def plot_training_results(self, save_path=None):
        """Plot training results."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Actor-Critic Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot moving average
        if len(self.episode_rewards) > 100:
            moving_avg = [np.mean(self.episode_rewards[max(0, i-100):i+1]) 
                         for i in range(len(self.episode_rewards))]
            ax1.plot(moving_avg, 'r-', label='Moving Average (100 episodes)')
            ax1.legend()
        
        # Plot actor losses
        if self.actor_losses:
            ax2.plot(self.actor_losses)
            ax2.set_title('Actor Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Actor Loss')
            ax2.grid(True)
        
        # Plot critic losses
        if self.critic_losses:
            ax3.plot(self.critic_losses)
            ax3.set_title('Critic Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Critic Loss')
            ax3.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
        return fig 