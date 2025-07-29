import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """Deep Q-Network Agent."""
    
    def __init__(self, state_size, action_size, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 buffer_size=10000, batch_size=32, target_update=100):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training metrics
        self.losses = []
        self.rewards = []
        self.episode_rewards = []
        self.update_count = 0
        
        # Update target network
        self.update_target_network()
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy."""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state)
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer."""
        self.memory.push(state, action, reward, next_state, done)
    
    def replay(self):
        """Train the network on a batch of experiences."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update == 0:
            self.update_target_network()
        
        # Store loss
        self.losses.append(loss.item())
    
    def train(self, env, episodes=1000, max_steps=200):
        """Train the DQN agent."""
        print("Training DQN Agent...")
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                self.remember(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                
                self.replay()
                
                if done:
                    break
            
            self.episode_rewards.append(episode_reward)
            
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        print("DQN Training completed!")
        return self.episode_rewards
    
    def evaluate(self, env, episodes=10, render=False):
        """Evaluate the trained agent."""
        print("Evaluating DQN Agent...")
        
        total_rewards = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            for step in range(200):
                action = self.act(state, training=False)
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
        
        print(f"DQN Evaluation Results:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        
        return total_rewards, avg_reward, std_reward
    
    def save_model(self, filepath):
        """Save the trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'losses': self.losses
        }, filepath)
        print(f"DQN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.losses = checkpoint['losses']
        print(f"DQN model loaded from {filepath}")
    
    def plot_training_results(self, save_path=None):
        """Plot training results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot episode rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('DQN Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.grid(True)
        
        # Plot moving average
        if len(self.episode_rewards) > 100:
            moving_avg = [np.mean(self.episode_rewards[max(0, i-100):i+1]) 
                         for i in range(len(self.episode_rewards))]
            ax1.plot(moving_avg, 'r-', label='Moving Average (100 episodes)')
            ax1.legend()
        
        # Plot losses
        if self.losses:
            ax2.plot(self.losses)
            ax2.set_title('DQN Training Loss')
            ax2.set_xlabel('Training Step')
            ax2.set_ylabel('Loss')
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training plots saved to {save_path}")
        
        plt.show()
        return fig


from environment.custom_env import LiberianEntrepreneurshipEnv

