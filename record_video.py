import gym
import torch
from gym.wrappers import RecordVideo
from policy_net import PolicyNet

def record(model_path="cartpole_policy.pth", output_dir="videos"):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder=output_dir, name_prefix="cartpole-agent")

    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNet(obs_dim, 128, action_dim)
    policy.load_state_dict(torch.load(model_path, map_location="cpu"))
    policy.eval()

    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = policy(state_v)
        action = torch.argmax(probs, dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"Recorded video, reward={total_reward}")
    env.close()

if __name__ == "__main__":
    record()
