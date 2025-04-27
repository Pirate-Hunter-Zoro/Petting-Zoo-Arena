from magent2.environments import battle_v4
from actor_critic import ActorCritic
from ppo_update import ppo_update
from ppo_buffer import RolloutBuffer
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import math
import matplotlib.pyplot as plt
import os

env = battle_v4.parallel_env()

# For when we save models
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

# Setup
obs_shape = env.observation_space(env.agents[0]).shape
obs_dim = np.prod(obs_shape)  # <-- flatten properly
act_dim = env.action_space(env.agents[0]).n
policy = ActorCritic(obs_dim, act_dim)
buffer = RolloutBuffer()

# We are going to be updating the actor and critic separately, so we need to split the parameters
actor_params = list(policy.actor.parameters()) + list(policy.shared.parameters())
critic_params = list(policy.critic.parameters()) + list(policy.shared.parameters())

optimizer_actor = optim.Adam(actor_params, lr=1e-4) # Smaller RL for actor
optimizer_critic = optim.Adam(critic_params, lr=3e-4) # Bigger RL for critic


total_episodes = 5000

reward_history = []
episode_lengths = []
smoothed_rewards = deque(maxlen=10)  # Last 10 episodes
smoothed_lengths = deque(maxlen=10)

"""
We do this:

Reset environment.

Loop:

    Build a dict of actions for all alive agents.

    Step all actions at once.

    Store (obs, action, log_prob, reward, done, value) for all agents into the buffer.

    Continue until batch_size steps are collected (or the episode ends).

    Then call PPO update after collecting enough data.
"""
batch_size = 2048  # How many total steps before we train
mini_batch_size = 64  # For PPO updates
ppo_epochs = 10  # How many times we update per batch

initial_ent_coef = 0.02
final_ent_coef = 0.0001
decay_episodes = 500  # Number of episodes over which to decay

for episode in range(total_episodes):
    episode_reward_total = 0
    episode_step_count = 0
    observations = env.reset()
    terminated = {agent: False for agent in env.agents}

    # Decay entropy coefficient
    progress = min(episode / decay_episodes, 1.0)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))  # Cosine from 1 ➔ 0 smoothly
    ent_coef = final_ent_coef + (initial_ent_coef - final_ent_coef) * cosine_decay

    while len(buffer) < batch_size:
        actions = {}
        values = {}
        log_probs = {}

        for agent, obs in observations.items():
            if terminated[agent]:
                continue
            obs_tensor = torch.tensor(obs, dtype=torch.float32).flatten().unsqueeze(0)
            action_val, log_prob, value = policy.get_action(obs_tensor)
            actions[agent] = action_val
            log_probs[agent] = log_prob
            values[agent] = value

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            done = terminations[agent] or truncations[agent]
            reward = rewards[agent]
            
            # Survival reward (small bonus if alive)
            if not done:
                reward += 0.01

            # HACKY battle reward shaping based only on reward size
            if rewards[agent] >= 1.0:
                reward += 3.0  # Big reward (likely a kill or huge damage)
            elif rewards[agent] >= 0.5:
                reward += 1.0  # Moderate reward (likely a hit or assist)
            
            # Store transition into buffer
            if agent in observations:
                obs_tensor = torch.tensor(observations[agent], dtype=torch.float32).flatten()
                buffer.store(obs_tensor, actions[agent], log_probs[agent], reward, done, values[agent])
            
            episode_reward_total += reward
            episode_step_count += 1

        observations = next_obs
        terminated.update(terminations)
        
        if all(terminated.values()):
            break  # Episode ended, but maybe not enough batch yet — next episode will resume collection

    # Once we have enough data for the batch, train
    # Update policy
    buffer.compute_returns_and_advantages()
    ppo_update(policy, optimizer_actor, optimizer_critic, buffer, ent_coef=ent_coef, critic_epochs=2, mini_batch_size=64, epochs=4)
    reward_history.append(episode_reward_total)
    episode_lengths.append(episode_step_count)
    smoothed_rewards.append(episode_reward_total)
    smoothed_lengths.append(episode_step_count)

    # Every 200 episodes, print a little summary and visualize the agents
    if episode % 200 == 0:
        print(f"[Episode {episode}] "
            f"Avg Reward (last 10): {np.mean(smoothed_rewards):.2f}, "
            f"Avg Length: {np.mean(smoothed_lengths):.2f}")
        env.render()

    # Save the model every 500 episodes
    if (episode + 1) % 500 == 0:
        save_path = os.path.join(save_dir, f"policy_checkpoint_episode_{episode+1}.pth")
        torch.save(policy.state_dict(), save_path)
        print(f"Saved checkpoint: {save_path}")

    buffer.clear()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(reward_history, label='Total Reward')
plt.plot(np.convolve(reward_history, np.ones(10)/10, mode='valid'), label='Smoothed Reward')
plt.title('Total Reward per Episode')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(episode_lengths, label='Episode Length')
plt.plot(np.convolve(episode_lengths, np.ones(10)/10, mode='valid'), label='Smoothed Length')
plt.title('Episode Length per Episode')
plt.legend()

plt.savefig('training_progress.png')
