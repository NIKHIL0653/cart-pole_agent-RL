# CartPole Agent

A reinforcement learning project implementing a Deep Q-Network (DQN) agent to solve the classic CartPole balancing problem from OpenAI Gym. The agent learns to balance a pole on a moving cart by applying appropriate forces (left or right) through trial and error.

## 🎯 Project Overview

This project demonstrates the application of deep reinforcement learning techniques to solve a continuous control problem. The agent uses DQN with experience replay, epsilon-greedy exploration, and reward shaping to achieve stable pole balancing. The project includes both command-line training scripts and an interactive Pygame-based training interface.

## ✨ Features

- **Deep Q-Learning Implementation**: Custom DQN agent with target network and experience replay
- **Interactive Training Interface**: Real-time visualization using Pygame with live statistics
- **Multiple Training Approaches**: Both DQN and Policy Gradient implementations
- **Model Persistence**: Save and load trained models for inference
- **Video Recording**: Record gameplay videos of trained agents
- **Performance Monitoring**: Real-time tracking of rewards, losses, and Q-values

## 🛠 Tech Stack

- **Python 3.8+**: Core programming language
- **PyTorch**: Deep learning framework for neural networks
- **OpenAI Gym**: Reinforcement learning environment
- **Pygame**: Interactive visualization and UI
- **NumPy**: Numerical computations

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager

## 🚀 Installation

1. Clone or navigate to the project directory:
   ```bash
   cd cart-pole-agent
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 Usage

### Training the Agent

#### Command-Line Training
Run the standard training script:
```bash
python train.py
```
This will train the DQN agent and save the best model to `dqn_cartpole.pth`.

#### Interactive Training with Visualization
Launch the Pygame-based training interface:
```bash
python console_game.py
```
This provides a real-time view of the agent's learning progress with:
- Live cart-pole visualization
- Score tracking
- Performance statistics
- Automatic model saving when target score (>150) is achieved

### Playing with Trained Agent

Load and run a trained model:
```bash
python play.py
```
This will run 5 episodes using the greedy policy from the saved model.

### Recording Gameplay Videos

Create videos of the agent's performance:
```bash
python record_video.py
```
Videos are saved to the `videos/` directory.

## 🧠 Key Concepts

### Deep Q-Learning (DQN)
- Uses a neural network to approximate the Q-value function
- Learns optimal action values for each state
- Balances exploration vs exploitation using epsilon-greedy strategy

### Experience Replay
- Stores past experiences in a replay buffer
- Samples random batches for training to break correlation between consecutive samples
- Improves learning stability and efficiency

### Double DQN
- Uses separate networks for action selection and evaluation
- Reduces overestimation of Q-values
- Improves training stability

### Reward Shaping
- Provides additional rewards for pole stability and cart positioning
- Encourages smoother control and longer balancing times
- Includes penalties for high velocities and pole falls

### Target Network
- Separate network for computing target Q-values
- Updated periodically to improve training stability
- Prevents moving target problem in Q-learning

## 📁 Project Structure

```
cart-pole-agent/
├── agent.py              # DQN agent implementation with training logic
├── console_game.py       # Interactive Pygame training interface
├── game.py               # Policy gradient training script
├── model.py              # DQN neural network architecture
├── play.py               # Inference script for trained models
├── record_video.py       # Video recording utility
├── replay_buffer.py      # Experience replay buffer implementation
├── train.py              # Command-line training script
├── requirements.txt      # Python dependencies
├── dqn_cartpole.pth      # Trained DQN model (generated)
├── cartpole_policy.pth   # Trained policy model (generated)
└── __pycache__/          # Python bytecode cache
```

## 🎯 Training Parameters

Key hyperparameters used in training:
- **Learning Rate**: 5e-4 (agent.py), 1e-3 (train.py)
- **Discount Factor (γ)**: 0.995
- **Batch Size**: 32-64
- **Replay Buffer Capacity**: 10,000-30,000
- **Target Network Update**: Every 2,000 steps
- **Epsilon Decay**: From 1.0 to 0.02 over 10,000-20,000 steps

## 📊 Performance Metrics

The agent is considered solved when it achieves:
- Average score > 195 over 100 consecutive episodes (Gym standard)
- Interactive mode target: Score > 150 in a single episode

## 🤝 Contributing

Feel free to fork this project and submit pull requests with improvements, bug fixes, or additional features.

## 📄 License

This project is open-source and available under the MIT License.

## 🔗 References

- [OpenAI Gym CartPole Environment](https://gym.openai.com/envs/CartPole-v1/)
- [Deep Q-Learning Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

*Built with ❤️ for learning reinforcement learning concepts*