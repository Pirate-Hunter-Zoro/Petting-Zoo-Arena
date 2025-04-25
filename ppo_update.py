import torch

def ppo_update(policy, optimizer, buffer, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, epochs=4, batch_size=1024):
    states, actions, log_probs_old, returns, advantages = buffer.get()

    for _ in range(epochs):
        indices = torch.randperm(states.size(0))
        for start in range(0, states.size(0), batch_size):
            # Create mini-batch based on indices
            end = start + batch_size
            mb_idx = indices[start:end]

            # Action probabilities and values for each state of the mini-batch
            logits, values = policy(states[mb_idx])
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - log_probs_old[mb_idx])
            surr1 = ratios * advantages[mb_idx]
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((returns[mb_idx] - values.squeeze()) ** 2).mean()

            loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
