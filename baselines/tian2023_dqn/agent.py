from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Sequential):
    def __init__(self):
        super().__init__(nn.Linear(180, 128), nn.ReLU(), nn.Linear(128, 64),
                         nn.ReLU(), nn.Linear(64, 9))


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.states = np.empty((capacity, 180), dtype=np.float32)
        self.next_states = np.empty_like(self.states)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminals = np.empty(capacity, dtype=np.float32)
        self.capacity, self.size, self.position = capacity, 0, 0

    def add(self, state, action, reward, next_state, terminal):
        i = self.position
        self.states[i], self.next_states[i] = state, next_state
        self.actions[i], self.rewards[i], self.terminals[i] = action, reward, terminal
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng):
        indices = rng.choice(self.size, size=batch_size, replace=False)
        return tuple(x[indices] for x in (self.states, self.actions, self.rewards,
                                          self.next_states, self.terminals))


def resolve_device(name: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if name == "auto" else torch.device(name)


def dqn_targets(rewards, terminals, next_q, gamma):
    # Vanilla DQN: target network supplies both the maximum and its value.
    return rewards + gamma * (1 - terminals) * next_q.max(dim=1).values


class DQNAgent:
    def __init__(self, config: dict, device="auto"):
        self.config = config
        self.device = resolve_device(device)
        self.online = QNetwork().to(self.device)
        self.target = QNetwork().to(self.device)
        self.sync_target()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config["learning_rate"])
        self.updates = 0

    @torch.no_grad()
    def act(self, state, epsilon, rng):
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.integers(9))
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return int(self.online(x).argmax(dim=1).item())

    def sync_target(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    def update(self, replay, rng):
        state, action, reward, next_state, terminal = [
            torch.as_tensor(x, device=self.device)
            for x in replay.sample(self.config["batch_size"], rng)
        ]
        with torch.no_grad():
            expected = dqn_targets(reward, terminal, self.target(next_state), self.config["gamma"])
        predicted = self.online(state).gather(1, action[:, None]).squeeze(1)
        loss = F.smooth_l1_loss(predicted, expected)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite DQN loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.config["max_grad_norm"], error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())
