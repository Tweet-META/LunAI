from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from rl.legacy.replay_buffer import ReplayBatch


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int = 9
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    target_update_interval: int = 1000
    device: str = "auto"


class QNetwork(nn.Module):
    # Create a simple MLP Q-network.
    def __init__(self, state_dim: int, action_dim: int, hidden_dim_1: int, hidden_dim_2: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, action_dim),
        )

    # Return Q-values for all actions.
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.net(states)


class DQNAgent:
    # Create online and target Q-networks.
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = self._select_device(config.device)
        self.online_net = QNetwork(config.state_dim, config.action_dim, config.hidden_dim_1, config.hidden_dim_2).to(self.device)
        self.target_net = QNetwork(config.state_dim, config.action_dim, config.hidden_dim_1, config.hidden_dim_2).to(self.device)
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self.train_steps = 0
        self.update_target()

    # Choose CPU or CUDA for training.
    def _select_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    # Choose an action with epsilon-greedy exploration.
    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if np.random.random() < epsilon:
            return int(np.random.randint(0, self.config.action_dim))
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    # Return Q-values for debugging a single state.
    def action_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_tensor).squeeze(0)
        return q_values.detach().cpu().numpy()

    # Run one DQN gradient update and return the loss.
    def update(self, batch: ReplayBatch) -> float:
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online_net(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.config.gamma * (1.0 - dones) * next_q_values

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.config.target_update_interval == 0:
            self.update_target()
        return float(loss.item())

    # Copy online network weights into the target network.
    def update_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save the online network and training metadata.
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.online_net.state_dict(),
                "target_model": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "train_steps": self.train_steps,
            },
            path,
        )

    # Load network weights and optimizer state.
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["model"])
        self.target_net.load_state_dict(checkpoint["target_model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))


# Load a DQN config dictionary from a checkpoint.
def load_dqn_config(path: str, device: str = "auto") -> DQNConfig:
    checkpoint = torch.load(path, map_location="cpu")
    config_data = dict(checkpoint["config"])
    config_data["device"] = device
    return DQNConfig(**config_data)
