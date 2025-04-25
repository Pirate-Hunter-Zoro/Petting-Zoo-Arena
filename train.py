from pettingzoo.magent import battle_v4
from actor_critic import ActorCritic
from ppo_update import ppo_update
from ppo_buffer import RolloutBuffer
import torch

env = battle_v4.env()
env.reset()

# Assumes all agents share the same policy
obs_dim = env.observation_space(env.agents[0]).shape[0]
act_dim = env.action_space(env.agents[0]).n

policy = ActorCritic(obs_dim, act_dim)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
buffer = RolloutBuffer()

total_episodes = 500

for episode in range(total_episodes):
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, _ = env.last()
        done = termination or truncation

        if done:
            action = None
            log_prob = torch.tensor(0.0)
            value = torch.tensor(0.0)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action_val, log_prob, value = policy.get_action(obs_tensor)
            action = action_val

        env.step(action)

        if obs is not None:
            buffer.store(obs_tensor.squeeze(), action or 0, log_prob, reward, done, value)

    buffer.compute_returns_and_advantages()
    ppo_update(policy, optimizer, buffer)
    buffer.clear()

    print(f"Episode {episode} done.")
