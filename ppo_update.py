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

            # First forward pass (for critic)
            values = policy.critic(policy.shared(states[mb_idx])).squeeze()

            # Critic loss
            value_loss = ((returns[mb_idx] - values) ** 2).mean()

            for _ in range(critic_epochs):
                optimizer_critic.zero_grad()
                # Run the model forward again to get the shared features with the computation graph
                shared_features = policy.shared(states[mb_idx])
                values = policy.critic(shared_features).squeeze()
                value_loss = ((returns[mb_idx] - values) ** 2).mean()

                value_loss.backward()
                optimizer_critic.step()

            # NEW forward pass (for actor) - otherwise the computation graph is broken
            shared_features = policy.shared(states[mb_idx])
            logits = policy.actor(shared_features)

            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()

            ratios = torch.exp(new_log_probs - log_probs_old[mb_idx])
            surr1 = ratios * advantages[mb_idx]
            surr2 = torch.clamp(ratios, 1 - clip_eps, 1 + clip_eps) * advantages[mb_idx]
            actor_loss = -torch.min(surr1, surr2).mean()

            optimizer_actor.zero_grad()
            loss = actor_loss - ent_coef * entropy
            loss.backward()
            optimizer_actor.step()