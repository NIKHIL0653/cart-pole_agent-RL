import gym
import torch
import numpy as np
import os
from agent import DQNAgent

def train(
    env_name="CartPole-v1",
    episodes=1000,
    max_steps=500,
    gamma=0.99,
    lr=1e-3,
    batch_size=64,
    buffer_capacity=20000,
    start_train=1000,
    target_update_steps=1000,
    save_path="dqn_cartpole.pth"
):
    env = gym.make(env_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        obs_dim, action_dim, device,
        lr=lr, gamma=gamma, batch_size=batch_size,
        buffer_capacity=buffer_capacity,
        start_train=start_train,
        target_update_steps=target_update_steps
    )

    epsilon_start = 1.0
    epsilon_final = 0.02
    epsilon_decay = 20000  # steps

    total_steps = 0
    scores = []
    best_avg = -float("inf")

    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        ep_reward = 0

        for t in range(max_steps):
            # linear epsilon decay by steps
            epsilon = epsilon_final + (epsilon_start - epsilon_final) * max(0, (epsilon_decay - total_steps)) / epsilon_decay
            action = agent.select_action(state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # small penalty on failure to encourage longer balancing
            r = reward
            if done and t < max_steps - 1:
                r = -10.0

            agent.push_transition(state, action, r, next_state, float(done))
            loss = agent.train_step()  # train per step

            state = next_state
            ep_reward += reward
            total_steps += 1

            if done:
                break

        scores.append(ep_reward)
        avg_last_100 = np.mean(scores[-100:])
        print(f"Episode {ep:4d} | Reward: {ep_reward:3.0f} | Avg100: {avg_last_100:6.2f} | Epsilon: {epsilon:.3f}")

        # Save best model
        if avg_last_100 > best_avg and ep >= 100:
            best_avg = avg_last_100
            agent.save(save_path)
            print(f"--> New best avg100 {best_avg:.2f}. Model saved to {save_path}")

        # early stop if solved
        if avg_last_100 >= 195.0:
            print(f"Solved in {ep} episodes! Avg100: {avg_last_100:.2f}")
            agent.save(save_path)
            break

    env.close()
    # final save
    if not os.path.exists(save_path):
        agent.save(save_path)
    print("Training finished.")

if __name__ == "__main__":
    train()
