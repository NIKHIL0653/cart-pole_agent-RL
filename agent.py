import random
import numpy as np
import torch
import torch.nn.functional as F
from model import DQN
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(
        self,
        obs_dim,
        action_dim,
        device,
        lr=5e-4,  # Reduced learning rate for stability
        gamma=0.995,  # Higher discount factor
        batch_size=32,  # Smaller batch size
        buffer_capacity=10000,  # Shorter buffer for more recent experiences
        start_train=1000,
        target_update_steps=2000  # Less frequent target updates
    ):
        self.device = device
        self.policy_net = DQN(obs_dim, action_dim).to(self.device)
        self.target_net = DQN(obs_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self.replay = ReplayBuffer(capacity=buffer_capacity)
        self.gamma = gamma
        self.batch_size = batch_size
        self.action_dim = action_dim

        self.steps_done = 0
        self.start_train = start_train
        self.target_update_steps = target_update_steps
        
        # For logging and diagnostics
        self.recent_losses = []
        self.q_value_history = []

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        state_v = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_v)
            # Store Q-values for diagnostics
            self.q_value_history.append(q_values.cpu().numpy().flatten())
            if len(self.q_value_history) > 100:  # Keep only recent 100
                self.q_value_history.pop(0)
            return int(q_values.argmax(dim=1).item())

    def get_shaped_reward(self, state, next_state, reward, done, step, max_steps):
        """Improved reward shaping for better stability"""
        if done and step < max_steps - 1:
            return -100.0  # Strong penalty for falling
        
        # Stability bonus based on pole angle and cart position
        cart_pos, cart_vel, pole_angle, pole_vel = next_state
        
        # Reward for keeping pole upright
        angle_reward = max(0, 1.0 - abs(pole_angle) / 0.3) * 0.5
        
        # Reward for staying near center
        position_reward = max(0, 1.0 - abs(cart_pos) / 2.4) * 0.3
        
        # Small penalty for high velocities (encourages smooth control)
        velocity_penalty = -0.1 * (abs(cart_vel) + abs(pole_vel)) / 10.0
        
        return reward + angle_reward + position_reward + velocity_penalty

    def push_transition(self, s, a, r, s2, done):
        self.replay.push(s, a, r, s2, done)
        self.steps_done += 1

    def train_step(self):
        if len(self.replay) < self.start_train:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device).unsqueeze(1)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)

        # Double DQN: Use policy network to select action, target network to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Clip Q-values to prevent instability
        q_values = q_values.clamp(-10, 10)
        target_q = target_q.clamp(-10, 10)

        loss = F.mse_loss(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()

        # Store loss for diagnostics
        self.recent_losses.append(loss.item())
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        # Update target network less frequently
        if self.steps_done % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def get_diagnostics(self):
        """Return diagnostic information for monitoring"""
        diagnostics = {}
        if self.recent_losses:
            diagnostics['avg_loss'] = np.mean(self.recent_losses)
        if self.q_value_history:
            recent_q = np.array(self.q_value_history[-10:])  # Last 10 Q-value sets
            diagnostics['avg_q_values'] = np.mean(recent_q, axis=0)
            diagnostics['q_value_std'] = np.std(recent_q, axis=0)
        return diagnostics

    def save(self, path="dqn_cartpole.pth"):
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, path)

    def load(self, path="dqn_cartpole.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'steps_done' in checkpoint:
            self.steps_done = checkpoint['steps_done']