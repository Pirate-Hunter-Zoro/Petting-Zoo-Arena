# === Imports ===
import os
import random
import torch
import matplotlib.pyplot as plt
import numpy as np

# === Local Modules ===
from magent2.environments import battle_v4
from actor_critic import ActorCritic
from ppo_update import ppo_update
from ppo_buffer import RolloutBuffer
from utils_env import get_device, save_checkpoint, load_frozen_policy
from rollout_logic import collect_experience

# === Hyperparameters ===
total_episodes = 5000
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

SAVE_POINT = 50                # Save current policy every 50 episodes
FROZEN_POLICY_POINT = 25       # Save frozen (self-play) policy every 25 episodes

batch_size = 8192
mini_batch_size = 256
ppo_epochs = 10
critic_epochs = 5

initial_ent_coef = 0.04
final_ent_coef = 0.001
decay_episodes = 2000

# === Device & Environment Setup ===
device = get_device()
env = battle_v4.parallel_env()
observations = env.reset()

agent = env.agents[0]
obs_shape = env.observation_space(agent).shape
act_dim = env.action_space(agent).n

# === Model & Optimization Setup ===
policy = ActorCritic(obs_shape, act_dim).to(device)
buffer = RolloutBuffer()

actor_params = list(policy.actor.parameters()) + list(policy.feature_extractor.parameters())
critic_params = list(policy.critic.parameters()) + list(policy.feature_extractor.parameters())

optimizer_actor = torch.optim.Adam(actor_params, lr=1e-4)
optimizer_critic = torch.optim.Adam(critic_params, lr=3e-4)
scheduler_actor = torch.optim.lr_scheduler.StepLR(optimizer_actor, step_size=5000, gamma=0.95)
scheduler_critic = torch.optim.lr_scheduler.StepLR(optimizer_critic, step_size=5000, gamma=0.95)

# === Stats Tracking ===
reward_history, episode_lengths = [], []
smoothed_rewards, smoothed_lengths = [], []

# === Preload Frozen Policies (if any exist) ===
frozen_policies = [file for file in os.listdir(save_dir) if file.startswith("frozen_policy")]

# === Main Training Loop ===
for episode in range(total_episodes):
    # Entropy coefficient decay schedule
    progress = min(episode / decay_episodes, 1.0)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
    ent_coef = final_ent_coef + (initial_ent_coef - final_ent_coef) * cosine_decay

    # Load frozen policy for self-play with 50% chance
    use_frozen = False
    frozen_policy = None
    if len(frozen_policies)>0 and random.random() < 0.5:
        chosen_path = os.path.join(save_dir, random.choice(frozen_policies))
        frozen_policy = load_frozen_policy(chosen_path, obs_shape, act_dim)
        use_frozen = True

    # Rollout experience collection
    total_reward, ep_len = collect_experience(
        env, policy, frozen_policy, use_frozen, buffer, device,
        batch_size, action_repeat=5
    )

    # PPO update
    buffer.compute_returns_and_advantages()
    buffer.returns = (buffer.returns - buffer.returns.mean()) / (buffer.returns.std() + 1e-8)
    ppo_update(policy, optimizer_actor, optimizer_critic, buffer, ent_coef=ent_coef,
               critic_epochs=critic_epochs, mini_batch_size=mini_batch_size, epochs=ppo_epochs)
    # Update the learning rate of the actor and critic
    scheduler_actor.step()
    scheduler_critic.step()

    # Stats tracking
    reward_history.append(total_reward)
    episode_lengths.append(ep_len)
    smoothed_rewards.append(np.mean(reward_history[-10:]))
    smoothed_lengths.append(np.mean(episode_lengths[-10:]))

    print(f"[Episode {episode}] Total Reward: {total_reward:.2f}, Length: {ep_len}, Entropy Coef: {ent_coef:.4f}")

    # Save current model
    if (episode + 1) % SAVE_POINT == 0:
        save_checkpoint(policy, save_dir, f"policy_checkpoint_episode_{episode+1}.pth")
    if (episode + 1) % FROZEN_POLICY_POINT == 0:
        save_checkpoint(policy, save_dir, f"frozen_policy_episode_{episode+1}.pth")
        frozen_policies = [file for file in os.listdir(save_dir) if file.startswith("frozen_policy")]

    buffer.clear()

# === Plot Results ===
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(reward_history, label='Total Reward')
plt.plot(smoothed_rewards, label='Smoothed Reward')
plt.legend()
plt.title('Reward per Episode')

plt.subplot(1, 2, 2)
plt.plot(episode_lengths, label='Episode Length')
plt.plot(smoothed_lengths, label='Smoothed Length')
plt.legend()
plt.title('Episode Length per Episode')

plt.savefig('training_progress.png')
