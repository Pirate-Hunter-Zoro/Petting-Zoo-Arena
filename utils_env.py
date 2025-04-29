import torch
import os
from actor_critic import ActorCritic

# Get the current device (GPU if available, otherwise CPU)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Save model checkpoint
def save_checkpoint(policy, save_dir, name):
    path = os.path.join(save_dir, name)
    torch.save(policy.state_dict(), path)
    print(f"Saved policy: {path}")

# Load frozen model checkpoint (for self-play)
def load_frozen_policy(path, obs_shape, act_dim):
    model = ActorCritic(obs_shape, act_dim)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
