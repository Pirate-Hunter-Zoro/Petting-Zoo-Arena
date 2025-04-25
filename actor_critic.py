import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )
        self.actor = nn.Linear(128, action_dim) # Action logits - the probability of each action given the state
        self.critic = nn.Linear(128, 1) # Value output - the actual estimated value of the state

    def forward(self, x):
        shared = self.shared(x)
        return self.actor(shared), self.critic(shared)

    def get_action(self, state):
        logits, value = self.forward(state)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value
