import gym
import torch
import torch.optim as optim
import numpy as np
import pygame
import sys
from model import PolicyNetwork

def select_action(policy, state):
    state = torch.FloatTensor(state).unsqueeze(0)
    probs = policy(state)
    action = torch.multinomial(probs, 1).item()
    return action, torch.log(probs[0, action])

def run_game():
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(obs_dim, action_dim)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)

    gamma = 0.99
    episode_rewards = []

    # --- Pygame Setup ---
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    pygame.display.set_caption("CartPole RL Agent Training")
    font = pygame.font.SysFont("Arial", 20)
    clock = pygame.time.Clock()

    running = True
    while running:
        state, _ = env.reset()
        log_probs, rewards = [], []
        score, done = 0, False

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running, done = False, True
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running, done = False, True

            action, log_prob = select_action(policy, state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            score += reward
            state = next_state
            done = terminated or truncated

            frame = env.render()
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(pygame.transform.scale(surf, (600, 350)), (0, 0))

            score_text = font.render(
                f"Episode {len(episode_rewards)+1} | Score: {score} | Avg(10): {np.mean(episode_rewards[-10:]):.2f}" 
                if episode_rewards else f"Episode {len(episode_rewards)+1} | Score: {score}",
                True, (255, 255, 255)
            )
            screen.blit(score_text, (20, 360))

            pygame.display.flip()
            clock.tick(60)

        # --- Training Step ---
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for log_prob, R in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * R)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        episode_rewards.append(score)

    env.close()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_game()
