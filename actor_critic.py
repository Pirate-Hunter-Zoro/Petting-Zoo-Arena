import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()

        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, 3, stride=2, padding=1),  # <-- FIXED
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_action(self, obs):
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate(self, obs, action):
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value
