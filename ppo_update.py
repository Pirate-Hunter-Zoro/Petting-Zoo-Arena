import torch

def ppo_update(policy, optimizer_critic, optimizer_actor, buffer, critic_epochs=2, clip_eps=0.2, ent_coef=0.0005, epochs=4, mini_batch_size=64):
    states, actions, log_probs_old, returns, advantages = buffer.get()

    states = states.detach()
    actions = actions.detach()
    log_probs_old = log_probs_old.detach()
    returns = returns.detach()
    advantages = advantages.detach()
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        indices = torch.randperm(states.size(0))
        for start in range(0, states.size(0), mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            # Value calculation for state
            features = policy.feature_extractor(states[mb_idx].squeeze(1))
            values = policy.critic(features).squeeze()

            # Critic loss
            value_loss = ((returns[mb_idx] - values) ** 2).mean()

            for _ in range(critic_epochs):
                optimizer_critic.zero_grad()
                # Run the model forward again to get the shared features with the computation graph
                shared_features = policy.feature_extractor(states[mb_idx])
                values = policy.critic(shared_features).squeeze()
                value_loss = ((returns[mb_idx] - values) ** 2).mean()

                value_loss.backward()
                optimizer_critic.step()

            # NEW forward pass (for actor) - otherwise the computation graph is broken
            shared_features = policy.feature_extractor(states[mb_idx])
            logits = policy.actor(shared_features)

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()

            # Compute actor loss
            ratios = torch.exp(new_log_probs - log_probs_old[mb_idx]) # How much more likely each action became after policy update
            surr1 = ratios * advantages[mb_idx] # How much better the each action was compared to expected baseline WEIGHTED by the likelihood ratio
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages[mb_idx] # E.g. if clip_eps = 0.2, then this is the range [0.8, 1.2] of the ratio, and update can cause a probability change of more than 20%
            # Maximizing the surrogate reward is equivalent to minimizing the negative surrogate reward
            actor_loss = -torch.min(surr1, surr2).mean() # If sometimes surr1 will be greater than surr2 or vice versa, then we will use the smaller one to avoid large policy updates

            optimizer_actor.zero_grad()
            loss = actor_loss - ent_coef * entropy
            # Add a soft KL penalty to avoid big policy shifts.
            kl_div = torch.distributions.kl_divergence(dist, torch.distributions.Categorical(logits=logits.detach())).mean()
            loss += 0.01 * kl_div
            loss.backward()
            optimizer_actor.step()