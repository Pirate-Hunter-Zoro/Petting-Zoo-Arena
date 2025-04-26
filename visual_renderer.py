import numpy as np
import matplotlib.pyplot as plt
import torch
from magent2.environments import battle_v4
from actor_critic import ActorCritic


# Create environment to get observation and action space info
env = battle_v4.parallel_env()
observations = env.reset()
done_agents = {agent: False for agent in env.agents}

# Pick any agent to sample an observation
sample_obs = list(observations.values())[0]

# Dynamically calculate state_dim
if isinstance(sample_obs, np.ndarray):
    state_dim = int(np.prod(sample_obs.shape))  # Flattened size
else:
    raise ValueError("Observation is not a numpy array!")

# Dynamically calculate action_dim
action_dim = env.action_space(env.agents[0]).n  # Assume discrete space

# Now initialize loading what we trained through ppo
policy = ActorCritic(state_dim=state_dim, action_dim=action_dim)
policy.load_state_dict(torch.load("checkpoints/policy_checkpoint_episode_5000.pth", map_location=torch.device("cpu")))
policy.eval()

# COLORS
TEAM_COLORS = {
    "red": (1.0, 0.3, 0.3),
    "blue": (0.3, 0.3, 1.0)
}

def render_arena_frame(env, ax, done_agents):
    ax.clear()
    ax.set_title("battle_v4 Parallel Arena View")

    # Grid size based on observation space
    sample_obs = next(iter(env.observation_spaces.values()))
    grid_height, grid_width, _ = sample_obs.shape
    ax.set_xlim(0, grid_width)
    ax.set_ylim(0, grid_height)
    ax.set_aspect('equal')

    # Draw agents as colored dots
    for agent in env.agents:
        if agent in done_agents and not done_agents[agent]:
            team = "red" if agent.startswith("red") else "blue"
            agent_idx = list(env.agents).index(agent)
            x = agent_idx % grid_width
            y = agent_idx // grid_width
            ax.plot(x + 0.5, y + 0.5, 'o', color=TEAM_COLORS[team], markersize=4)

    ax.set_xticks([])
    ax.set_yticks([])
    plt.pause(0.05)


def run_visual_parallel_episode(env, policy_fn, done_agents):
    observations = env.reset()

    _, ax = plt.subplots(figsize=(6, 6))
    done = False

    while not done:
        render_arena_frame(env, ax, done_agents)

        actions = {}
        for agent, obs in observations.items():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = policy_fn(obs_tensor)
            actions[agent] = action

        observations, _, terminations, truncations, _ = env.step(actions)
        for agent, terminated in terminations.items():
            if terminated:
                done_agents[agent] = True
        done = all(terminations.values()) or all(truncations.values())

    plt.show()


def ppo_policy(obs_tensor):
    if len(obs_tensor.shape) == 4:
        obs_tensor = obs_tensor.view(1, -1)
    action, _, _ = policy.get_action(obs_tensor)
    return action

agent_list = list(env.agents)
def random_policy(_):
    """Choose a random valid action."""
    action_space = env.action_space(agent_list[0])
    return action_space.sample()

if __name__ == "__main__":
    run_visual_parallel_episode(env, random_policy, done_agents)