# Group8 PPO Arena Agent

## Overview

This repository contains our custom Proximal Policy Optimization (PPO) agent designed for training on the PettingZoo `battle_v4` multi-agent environment. The agent uses a convolutional neural network (CNN) feature extractor and several modern reinforcement learning best practices to achieve strong competitive performance.

---

## Key Features

- **Convolutional Neural Network (CNN)** feature extractor for spatial observations
- **Frame Stacking (4 frames)** for richer temporal dynamics
- **Generalized Advantage Estimation (GAE)** for stable training
- **Advantage Normalization** to prevent gradient explosion
- **Entropy Regularization** with cosine decay for exploration
- **Batch Size** 64 (optimized for M1 Mac hardware)
- **Separate Actor and Critic networks**
- **Checkpoint saving** every 500 episodes
- **Reward smoothing and training visualization**

---

## Training Setup

- **Environment:** PettingZoo `battle_v4`, wrapped with `supersuit` frame stacking.
- **Actor Learning Rate:** 1e-4
- **Critic Learning Rate:** 3e-4
- **Optimizer:** Adam
- **Batch Size:** 64
- **PPO Epochs per Update:** 10
- **Clip Range:** 0.2
- **Initial Entropy Coefficient:** 0.01 (decaying to 0.001)
- **Frame Stack Depth:** 4
- **Observation Shape After Stacking:** (4, 18, 18)

---

## Files and Structure

| File | Purpose |
|:---|:---|
| `actor_critic.py` | Defines the CNN-based Actor-Critic architecture |
| `ppo_buffer.py` | Rollout buffer storing experiences and computing advantages |
| `ppo_update.py` | PPO policy and value updates with advantage normalization |
| `train.py` | Full training loop, logging, model saving, reward tracking |
| `training_progress.png` | Training reward and episode length curves |

---

## Expected Behavior

- Early episodes: Highly random movement, slow reward gain.
- After ~5000 episodes: Clear learning of attack behavior, group movement, and survival strategies.
- Plateau between 15–25 average reward depending on training length and hyperparameters.

---

## Future Improvements

- Upgrade to larger batch sizes if memory allows (e.g., 128+).
- Experiment with additional reward shaping (e.g., team-based rewards).
- Add TensorBoard logging for real-time monitoring.

---

## Notes

- This agent is designed for **self-play** training against identical opponents.
- For evaluation against different or stronger enemies, retraining or fine-tuning may be necessary.

---

> *Developed by Mikey Ferguson and Omega 🍜✨*
