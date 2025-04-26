import torch
from pettingzoo.magent import battle_v4
from actor_critic import ActorCritic
import numpy as np
import time

# Load environment
env = battle_v4.parallel_env()
env.reset()

# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate your policy (must match architecture used during training!)
observation_space = np.prod(env.observation_space(env.agents[0]).shape)
action_space = env.action_space(env.agents[0]).n
policy = ActorCritic(observation_space, action_space).to(device)

# Load model checkpoint
checkpoint_path = "checkpoints/policy_checkpoint_episode_500.pth"  # <-- Change this path if needed
policy.load_state_dict(torch.load(checkpoint_path, map_location=device))
policy.eval()

print(f"Loaded model from {checkpoint_path}")

# Evaluation loop
num_episodes = 5  # Number of episodes you want to watch
for ep in range(num_episodes):
    observations = env.reset()
    terminated = {agent: False for agent in env.agents}
    print(f"\nEpisode {ep + 1}")

    while not all(terminated.values()):
        actions = {}
        for agent in env.agents:
            if not terminated[agent]:
                obs = torch.tensor(observations[agent], dtype=torch.float32).flatten().unsqueeze(0).to(device)
                logits, _ = policy(obs)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample().item()
                actions[agent] = action

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update termination flags
        for agent in env.agents:
            terminated[agent] = terminations[agent] or truncations[agent]

        # Render the environment
        env.render()
        input("Press Enter for next move...") # Let the user decide when the next move happens

print("Evaluation complete.")