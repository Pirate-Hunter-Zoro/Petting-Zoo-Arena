import torch
from pettingzoo.magent import battle_v4
from actor_critic import ActorCritic
import numpy as np
import time

# Load environment
env = battle_v4.parallel_env()
env.reset()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate policy
agent = env.agents[0]
obs_shape = env.observation_space(agent).shape
act_dim = env.action_space(agent).n
policy = ActorCritic(obs_shape, act_dim).to(device)

# Load model checkpoint
checkpoint_path = "checkpoints/policy_checkpoint_episode_500.pth"  # <-- Update if needed
policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
policy.eval()

print(f"Loaded model from {checkpoint_path}")

# Evaluation loop
num_episodes = 5
for ep in range(num_episodes):
    observations = env.reset()
    terminated = {agent: False for agent in env.agents}
    print(f"\nEpisode {ep + 1}")

    while not all(terminated.values()):
        actions = {}
        for agent in env.agents:
            if not terminated[agent]:
                obs_tensor = torch.tensor(observations / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)
                obs_tensor = torch.clamp(obs_tensor, 0.0, 1.0)
                obs_tensor = torch.clamp(obs_tensor, 0.0, 1.0) 
                action, _, _ = policy.get_action(obs_tensor)
                actions[agent] = action.item()

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            terminated[agent] = terminations[agent] or truncations[agent]

        env.render()
        input("Press Enter for next move...")

print("Evaluation complete.")