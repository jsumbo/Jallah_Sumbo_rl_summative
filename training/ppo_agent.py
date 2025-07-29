import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt

class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO algorithm."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, action_size)
        
        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        
        # Actor output (action probabilities)
        action_probs = F.softmax(self.actor(x), dim=-1)
        
        # Critic output (state value)
        state_value = self.critic(x)
        
        return action_probs, state_value

class PPOAgent:
    """Proximal Policy Optimization (PPO) Agent."""
    
    def __init__(self, state_size, action_size, lr=3e-4, gamma=0.99, 
                 eps_clip=0.2, k_epochs=4, c1=0.5, c2=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.c1 = c1  # Value function coefficient
        self.c2 = c2  # Entropy coefficient
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Old policy for PPO updates
        self.old_policy = ActorCriticNetwork(state_size, action_size).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Training metrics
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self.total_losses = []
        
        # Episode storage
        self.reset_episode()
    
    def reset_episode(self):
        """Reset episode storage."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def act(self, state):
        """Choose action using current policy."""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs, state_value = self.old_policy(state)
        
        # Create categorical distribution and sample action
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Store for training
        self.states.append(state.squeeze().cpu().numpy())
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(state_value.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """Store reward and done flag."""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value=0, lam=0.95):
        """Compute Generalized Advantage Estimation (GAE)."""
        advantages = []
        returns = []
        
        gae = 0
        for i in reversed(range(len(self.rewards))):
            if i == len(self.rewards) - 1:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = next_value
            else:
                next_non_terminal = 1.0 - self.dones[i]
                next_value = self.values[i + 1]
            
            delta = self.rewards[i] + self.gamma * next_value * next_non_terminal - self.values[i]
            gae = delta + self.gamma * lam * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[i])
        
        return advantages, returns
    
    def update_policy(self):
        """Update policy using PPO algorithm."""
        if not self.rewards:
            return
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae()
        
        # Convert to tensors
        states = torch.FloatTensor(self.states).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update for k epochs
        for _ in range(self.k_epochs):
            # Get current policy outputs
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # Calculate ratio
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # Actor loss (policy loss)
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (value loss)
            critic_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # Total loss
            total_loss = actor_loss + self.c1 * critic_loss - self.c2 * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
        
        # Update old policy
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        # Store losses
        self.actor_losses.append(actor_loss.item())
        self.critic_losses.append(critic_loss.item())
        self.total_losses.append(total_loss.item())
        
        # Reset episode storage
        self.reset_episode()
    
    def train(self, env, episodes=1000, max_steps=200):
        """Train the PPO agent."""
        print("Training PPO Agent...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            self.reset_episode()
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.store_transition(reward, done)
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            # Update policy at the end of episode
            self.update_policy()
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
        
        print("PPO Training completed!")
        return self.episode_rewards
    
    def evaluate(self, env, episodes=10, render=False):
        """Evaluate the trained agent."""
        print("Evaluating PPO Agent...")
        
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(200):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    action_probs, _ = self.policy(state_tensor)
                
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
        
        print(f"PPO Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return total_rewards, avg_reward, std_reward
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'old_policy_state_dict': self.old_policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'actor_losses': self.actor_losses,
            'critic_losses': self.critic_losses,
            'total_losses': self.total_losses
        }, filepath)
        print(f"PPO model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.old_policy.load_state_dict(checkpoint['old_policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint['episode_rewards']
        self.actor_losses = checkpoint['actor_losses']
        self.critic_losses = checkpoint['critic_losses']
        self.total_losses = checkpoint['total_losses']
        print(f"PPO model loaded from {filepath}")
    
    def plot_training_results(self, save_path=None):
        """Plot training results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('PPO Training Rewards')
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
            ax2.set_title('PPO Actor Loss')
            ax2.set_xlabel('Episode')
            ax2.set_ylabel('Actor Loss')
            ax2.grid(True)
        
        # Plot critic losses
        if self.critic_losses:
            ax3.plot(self.critic_losses)
            ax3.set_title('PPO Critic Loss')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Critic Loss')
            ax3.grid(True)
        
        # Plot total losses
        if self.total_losses:
            ax4.plot(self.total_losses)
            ax4.set_title('PPO Total Loss')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Total Loss')
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
        return fig 