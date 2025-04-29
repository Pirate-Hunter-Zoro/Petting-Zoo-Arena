import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, action_dim):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(obs_shape[2], 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),  # GroupNorm after deeper layer
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.1),        # New Dropout to prevent overfitting
            nn.AdaptiveAvgPool2d((1,1)),  # Pool down to (1,1) regardless of input size
            nn.Flatten()
        )

        # Calculate flatten size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])
            n_flatten = self.feature_extractor(dummy_input).shape[1]

        self.actor = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self.critic = nn.Sequential(
            nn.Linear(n_flatten, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self):
        raise NotImplementedError

    def get_action(self, obs_batch):
        features = self.feature_extractor(obs_batch)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        return actions, dist.log_prob(actions), dist.entropy()

    def evaluate(self, obs, action):
        features = self.feature_extractor(obs)
        logits = self.actor(features)
        dist = torch.distributions.Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(features).squeeze(-1)
        return log_prob, entropy, value
