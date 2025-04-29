import torch

# Collect rollout experience for one episode
def collect_experience(env, policy, frozen_policy, use_frozen, buffer, device,
                       batch_size, action_repeat=5):
    observations = env.reset()
    terminated = {agent: False for agent in env.agents}
    episode_reward_total = 0
    episode_step_count = 0

    steps_since_action = 0
    current_actions = {}
    values = {}
    log_probs = {}

    while len(buffer) < batch_size: # Note that this function does NOTHING if the buffer is full
        # Only choose new actions every 'action_repeat' steps
        if steps_since_action % action_repeat == 0:
            actions = {}
            obs_batch = []
            alive_agents = []

            for agent, obs in observations.items():
                if not terminated[agent]:
                    obs_tensor = torch.tensor(obs / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)
                    obs_tensor = torch.clamp(obs_tensor, 0.0, 1.0)
                    obs_batch.append(obs_tensor)
                    alive_agents.append(agent)

            if len(obs_batch) > 0:
                obs_batch = torch.stack(obs_batch)
                actions_batch = torch.zeros(len(alive_agents), dtype=torch.int64).to(device)
                log_probs_batch = torch.zeros(len(alive_agents)).to(device)
                values_batch = torch.zeros(len(alive_agents)).to(device)

                for idx in range(len(alive_agents)):
                    agent_obs = obs_batch[idx].unsqueeze(0)
                    if use_frozen and frozen_policy and idx % 2 == 0:
                        a, lp, v = frozen_policy.get_action(agent_obs)
                    else:
                        a, lp, v = policy.get_action(agent_obs)
                    actions_batch[idx] = a
                    log_probs_batch[idx] = lp
                    values_batch[idx] = v

                for idx, agent in enumerate(alive_agents):
                    actions[agent] = actions_batch[idx]
                    log_probs[agent] = log_probs_batch[idx]
                    values[agent] = values_batch[idx]

            current_actions = actions

        # Step the environment with chosen/repeated actions
        next_obs, rewards, terminations, truncations, _ = env.step(current_actions)

        for agent in env.agents:
            done = terminations[agent] or truncations[agent]
            reward = rewards[agent]

            # Reward shaping (survival bonus + kill bonus)
            if not done:
                reward += 0.02
            if rewards[agent] >= 1.0:
                reward += 3.0
            elif rewards[agent] >= 0.5:
                reward += 1.0

            if agent in observations:
                obs_tensor = torch.tensor(observations[agent] / 255.0, dtype=torch.float32).permute(2, 0, 1).to(device)
                obs_tensor = torch.clamp(obs_tensor, 0.0, 1.0)
                buffer.store(obs_tensor, current_actions[agent], log_probs[agent], reward, done, values[agent])

            episode_reward_total += reward
            episode_step_count += 1

        observations = next_obs
        terminated.update(terminations)
        steps_since_action += 1

        if all(terminated.values()):
            break

    return episode_reward_total, episode_step_count
