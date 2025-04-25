import torch

class RolloutBuffer:
    def __init__(self):
        # Initialize lists to store experience tuples
        self.states = []
        self.actions = []
        self.log_probs = []  # Log probabilities of actions taken (for PPO's ratio clipping)
        self.rewards = []
        self.dones = []      # Boolean flags indicating episode termination
        self.values = []     # Estimated state values (from the value function)

    def store(self, state, action, log_prob, reward, done, value):
        # Save one timestep of data into the buffer
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, gamma=0.99, lam=0.95):
        """
        Computes the advantage estimates using GAE (Generalized Advantage Estimation)
        and the returns for policy/value function updates.
        """
        returns = []
        advantages = []
        gae = 0  # GAE accumulator
        value_next = 0  # Assume terminal state's value is 0

        # Process buffer in reverse (from the last timestep to the first)
        for t in reversed(range(len(self.rewards))):
            # Temporal difference error (TD residual)
            delta = self.rewards[t] + gamma * value_next * (1 - self.dones[t]) - self.values[t]
            # GAE recursive formula
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)  # Prepend to preserve correct order
            value_next = self.values[t]
            returns.insert(0, gae + self.values[t])  # Add value to advantage to get return

        # Convert lists to torch tensors for use in training
        self.returns = torch.tensor(returns, dtype=torch.float32)
        self.advantages = torch.tensor(advantages, dtype=torch.float32)

    def get(self):
        # Return stacked/batched data for training
        return (
            torch.stack(self.states),  # Convert list of states into a tensor batch
            torch.tensor(self.actions),  # Actions as tensor
            torch.tensor(self.log_probs),  # Log probs as tensor
            self.returns,  # Precomputed returns
            self.advantages  # Precomputed advantages
        )

    def clear(self):
        # Reset buffer (like purging all cursed energy—clean slate)
        self.__init__()
