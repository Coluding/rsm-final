import torch
import numpy as np
import random
from collections import deque
from dataclasses import dataclass

@dataclass
class Step:
    state: np.ndarray
    action: int
    logp: float
    reward: float
    done: bool
    timestep: int
    value: float
    next_value: float
    advantage: float = 0.0
    returns: float = 0.0

class OffPolicyReplayBuffer:
    def __init__(self, capacity, priority=False):
        self.buffer = deque(maxlen=capacity)
        self.priority = priority
        self.priorities = deque(maxlen=capacity) if priority else None

    def push(self, state, action, reward, next_state, done, timestep, priority=1.0):
        self.buffer.append((state, action, reward, next_state, done, timestep))
        if self.priority:
            self.priorities.append(priority)

    def sample(self, batch_size):
        if self.priority:
            probabilities = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
            batch = [self.buffer[idx] for idx in indices]
        else:
            batch = random.sample(self.buffer, batch_size)
        return batch

    def update_priorities(self, indices, new_priorities):
        if self.priority:
            for idx, priority in zip(indices, new_priorities):
                self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)


class OnPolicyReplayBuffer:
    def __init__(self, capacity = 100000, ):
        from collections import defaultdict
        self.episodes = defaultdict(list)
        self._available_keys = set()
        self._current_key = None
        self.max_capacity = capacity # in episodes
        self.capacity = 0

    def push(self, state, action, log_prob, reward, next_state, done, timestep):
        if timestep == 1:
            if len(self.episodes) >= self.max_capacity:
                self.episodes.popitem()
            self._current_key = self._create_episode_key()

        self.episodes[self._current_key].append((state, action, log_prob, reward, next_state, done, timestep))
        self.capacity += 1

        if done:
            self._available_keys.add(self._current_key)

    def _create_episode_key(self):
        from uuid import uuid4
        return uuid4()

    def __call__(self, batch_size):
        if len(self._available_keys) == 0:
            return None
        if batch_size > len(list(self.episodes.values())[0]):
            raise ValueError("Batch size is larger than transitions in one episode. Reduce batch size or use larger episodes.")
        while True:
            try:
                #TODO: this is not optimal. We can only sample episodes that are finished and have length >= batch_size because the whole code (advantage computation and PPO optimization) works with ordered batch items. That means the sampled elments must come from the same episode an dwe cannot merge episodes.
                episode_key = random.choice(list(self._available_keys))
                episode = self.episodes[episode_key]
                starting_point = random.randint(0, len(episode) - batch_size)
                batch = episode[starting_point:starting_point + batch_size]
            except:
                pass
            yield batch

    def clear(self):
        self.episodes.clear()

    def __len__(self):
        return self.capacity


class TrajectoryReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, trajectory):
        self.buffer.append(trajectory)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch

    def __len__(self):
        return len(self.buffer)

    def gen(self, batch_size, shuffle=True):
        random.shuffle(self.buffer) if shuffle else None
        for i in range(0, len(self.buffer), batch_size):
            yield self.buffer[i:i + batch_size]



class OnPolicyTrajectoryBuffer:
    """
    Collects complete episodes (trajectories), then computes GAE(λ) advantages
    and TD(λ) returns per episode. Afterwards, you can iterate flat minibatches
    with consistent (adv, returns, old_logp, value) for PPO.
    """
    def __init__(self, capacity_episodes=1000, gamma=0.99, gae_lambda=0.95):
        self.gamma = gamma
        self.lmbda = gae_lambda
        self.capacity = capacity_episodes
        self.episodes = deque(maxlen=capacity_episodes)
        self._current = []  # steps of ongoing episode

    def start_episode(self):
        self._current = []

    def push(self, state, action, logp, reward, done, timestep, value, next_value, *args):
        self._current.append(Step(state, action, float(logp), float(reward), bool(done),
                                  int(timestep), float(value), float(next_value)))
        if done:
            self.finish_episode()

    def finish_episode(self):
        if not self._current:
            return
        # Compute deltas, GAE, returns for this episode
        T = len(self._current)
        adv = [0.0] * T
        ret = [0.0] * T

        # δ_t = r_t + γ*(1-d_t)*V_{t+1} - V_t
        deltas = [
            s.reward + self.gamma * (0.0 if s.done else s.next_value) - s.value
            for s in self._current
        ]

        gae = 0.0
        for t in reversed(range(T)):
            mask = 0.0 if self._current[t].done else 1.0
            gae = deltas[t] + self.gamma * self.lmbda * mask * gae
            adv[t] = gae
            ret[t] = self._current[t].value + adv[t]  # TD(λ) target

        # Attach computed fields
        for i, s in enumerate(self._current):
            # Attach as attributes for later readout
            s.advantage = adv[i]
            s.returns = ret[i]

        self.episodes.append(self._current)
        self._current = []

    def clear(self):
        self.episodes.clear()
        self._current = []

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)

    def iter_minibatches(self, batch_size, shuffle=True, advantage_norm=True):
        """Flatten after computing GAE/returns and yield minibatches."""
        flat = [s for ep in self.episodes for s in ep]
        if shuffle:
            random.shuffle(flat)

        # Optional: normalize advantages across the flat batch
        if advantage_norm and flat:
            a = np.array([s.advantage for s in flat], dtype=np.float32)
            a = (a - a.mean()) / (a.std() + 1e-8)
            for i, s in enumerate(flat):
                s.advantage = float(a[i])

        for i in range(0, len(flat), batch_size):
            batch = flat[i:i + batch_size]
            yield {
                "states": [s.state for s in batch],
                "actions": torch.tensor([s.action for s in batch], dtype=torch.long),
                "old_logp": torch.tensor([s.logp for s in batch], dtype=torch.float32),
                "advantages": torch.tensor([s.advantage for s in batch], dtype=torch.float32),
                "returns": torch.tensor([s.returns for s in batch], dtype=torch.float32),
                "values_old": torch.tensor([s.value for s in batch], dtype=torch.float32),
                "dones": torch.tensor([s.done for s in batch], dtype=torch.float32),
                "timesteps": torch.tensor([s.timestep for s in batch], dtype=torch.long),
            }