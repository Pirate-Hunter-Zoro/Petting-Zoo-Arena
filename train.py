import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import math

from magent2.environments import battle_v4
from actor_critic import ActorCritic
from ppo_update import ppo_update
from ppo_buffer import RolloutBuffer

# Initialize environment and agent
env = battle_v4.parallel_env()
observations = env.reset()

agent = env.agents[0]
obs_shape = env.observation_space(agent).shape
act_dim = env.action_space(agent).n

policy = ActorCritic(obs_shape, act_dim)

save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)

buffer = RolloutBuffer()

actor_params = list(policy.actor.parameters()) + list(policy.feature_extractor.parameters())
critic_params = list(policy.critic.parameters()) + list(policy.feature_extractor.parameters())

optimizer_actor = optim.Adam(actor_params, lr=1e-4)
optimizer_critic = optim.Adam(critic_params, lr=3e-4)
# For learning rate decay
scheduler_actor = torch.optim.lr_scheduler.StepLR(optimizer_actor, step_size=5000, gamma=0.95)
scheduler_critic = torch.optim.lr_scheduler.StepLR(optimizer_critic, step_size=5000, gamma=0.95)

total_episodes = 5000

reward_history = []
episode_lengths = []
smoothed_rewards = []
smoothed_lengths = []

batch_size = 8192
mini_batch_size = 256
ppo_epochs = 10
critic_epochs = 5

initial_ent_coef = 0.04
final_ent_coef = 0.001
decay_episodes = 2000

for episode in range(total_episodes):
    episode_reward_total = 0
    episode_step_count = 0
    observations = env.reset()
    terminated = {agent: False for agent in env.agents}

    # Decay entropy coefficient
    progress = min(episode / decay_episodes, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    ent_coef = final_ent_coef + (initial_ent_coef - final_ent_coef) * cosine_decay

    while len(buffer) < batch_size:
        actions = {}
        values = {}
        log_probs = {}

        obs_batch = []
        alive_agents = []

        for agent, obs in observations.items():
            if not terminated[agent]:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                obs_batch.append(obs_tensor)
                alive_agents.append(agent)

        if len(obs_batch) > 0:
            obs_batch = torch.stack(obs_batch)  # Stack all alive agents
            actions_batch, log_probs_batch, values_batch = policy.get_action(obs_batch)

            for idx, agent in enumerate(alive_agents):
                actions[agent] = actions_batch[idx]
                log_probs[agent] = log_probs_batch[idx]
                values[agent] = values_batch[idx]

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            done = terminations[agent] or truncations[agent]
            reward = rewards[agent]

            # Survival reward (small bonus if alive)
            if not done:
                reward += 0.02

            # Reward shaping for actions
            if rewards[agent] >= 1.0:
                reward += 3.0
            elif rewards[agent] >= 0.5:
                reward += 1.0

            if agent in observations:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1)
                buffer.store(obs_tensor, actions[agent], log_probs[agent], reward, done, values[agent])

            episode_reward_total += reward
            episode_step_count += 1

        observations = next_obs
        terminated.update(terminations)

        if all(terminated.values()):
            break

    buffer.compute_returns_and_advantages()
    ppo_update(policy, optimizer_actor, optimizer_critic, buffer, ent_coef=ent_coef, critic_epochs=critic_epochs, mini_batch_size=mini_batch_size, epochs=ppo_epochs)
    # Learning rates
    scheduler_actor.step()
    scheduler_critic.step()

    reward_history.append(episode_reward_total)
    episode_lengths.append(episode_step_count)
    smoothed_rewards.append(np.mean(reward_history[-10:]))
    smoothed_lengths.append(np.mean(episode_lengths[-10:]))

    print(f"[Episode {episode}] Total Reward: {episode_reward_total:.2f}, Episode Length: {episode_step_count}, Entropy Coefficient: {ent_coef:.4f}")

    if episode % 200 == 0:
        env.render()

    if (episode + 1) % 50 == 0:
        save_path = os.path.join(save_dir, f"policy_checkpoint_episode_{episode+1}.pth")
        torch.save(policy.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

    buffer.clear()

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reward_history, label='Total Reward')
plt.plot(smoothed_rewards, label='Smoothed Reward')
plt.title('Total Reward per Episode')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(episode_lengths, label='Episode Length')
plt.plot(smoothed_lengths, label='Smoothed Length')
plt.title('Episode Length per Episode')
plt.legend()

plt.savefig('training_progress.png')