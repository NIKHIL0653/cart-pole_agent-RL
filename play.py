import gym
import torch
from agent import DQNAgent
import numpy as np

def play(model_path="dqn_cartpole.pth", episodes=5):
    env = gym.make("CartPole-v1", render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(obs_dim, action_dim, device)
    agent.load(model_path)
    agent.policy_net.eval()

    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, epsilon=0.0)  # greedy
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            ep_reward += reward
            env.render()
        print(f"Play Episode {ep+1} Reward: {ep_reward}")

    env.close()

if __name__ == "__main__":
    play()
