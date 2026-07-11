from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from rl.cnn_observation_utils import CNNObservation


@dataclass
class CNNPPOConfig:
    red_shape: tuple[int, int, int]
    yellow_shape: tuple[int, int, int]
    blue_shape: tuple[int, int, int]
    player_dim: int
    frame_stack: int = 1
    frame_stack_interval: int = 1
    action_dim: int = 9
    hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 1e-4
    clip_range: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 256
    target_kl: float = 0.03
    device: str = "auto"


class CNNActorCritic(nn.Module):
    # Create a multi-branch actor-critic network for map observations.
    def __init__(self, config: CNNPPOConfig):
        super().__init__()
        self.red_encoder = nn.Sequential(
            nn.Conv2d(config.red_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.yellow_encoder = nn.Sequential(
            nn.Conv2d(config.yellow_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.blue_encoder = nn.Sequential(
            nn.Conv2d(config.blue_shape[0], 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )
        self.player_encoder = nn.Sequential(
            nn.Linear(config.player_dim, 32),
            nn.ReLU(),
        )
        feature_dim = 32 * 4 * 4 + 16 * 2 * 2 + 16 * 2 * 2 + 32
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
        )
        self.actor = nn.Linear(config.hidden_dim // 2, config.action_dim)
        self.critic = nn.Linear(config.hidden_dim // 2, 1)

    # Return action logits and state value.
    def forward(self, states: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        red_features = self.red_encoder(states["red"])
        yellow_features = self.yellow_encoder(states["yellow"])
        blue_features = self.blue_encoder(states["blue"])
        player_features = self.player_encoder(states["player"])
        features = torch.cat([red_features, yellow_features, blue_features, player_features], dim=1)
        trunk_features = self.trunk(features)
        logits = self.actor(trunk_features)
        values = self.critic(trunk_features).squeeze(-1)
        return logits, values


class CNNPPOAgent:
    # Create the CNN PPO model and optimizer.
    def __init__(self, config: CNNPPOConfig):
        self.config = config
        self.device = self._select_device(config.device)
        self.model = CNNActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.train_steps = 0

    # Choose CPU or CUDA for training.
    def _select_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    # Convert numpy state arrays into tensors on the agent device.
    def _state_to_tensors(self, state: CNNObservation, batched: bool = False) -> dict[str, torch.Tensor]:
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in state.items()
        }
        if not batched:
            tensors = {key: value.unsqueeze(0) for key, value in tensors.items()}
        return tensors

    # Sample one action from the current policy.
    def select_action(self, state: CNNObservation) -> tuple[int, float, float]:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(state)
            logits, values = self.model(state_tensors)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(values.item())

    # Sample one action for every state in a batched observation.
    def select_actions_batch(self, states: CNNObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(states, batched=True)
            logits, values = self.model(state_tensors)
            distribution = Categorical(logits=logits)
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)
        return (
            actions.detach().cpu().numpy().astype(np.int64),
            log_probs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )

    # Choose the highest-probability action for evaluation.
    def select_greedy_action(self, state: CNNObservation) -> int:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(state)
            logits, _ = self.model(state_tensors)
        return int(torch.argmax(logits, dim=1).item())

    # Return action probabilities for debugging.
    def action_probs(self, state: CNNObservation) -> np.ndarray:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(state)
            logits, _ = self.model(state_tensors)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.detach().cpu().numpy()

    # Return the critic value for one state.
    def state_value(self, state: CNNObservation) -> float:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(state)
            _, value = self.model(state_tensors)
        return float(value.item())

    # Return one value estimate for every state in a batched observation.
    def state_values_batch(self, states: CNNObservation) -> np.ndarray:
        with torch.no_grad():
            state_tensors = self._state_to_tensors(states, batched=True)
            _, values = self.model(state_tensors)
        return values.detach().cpu().numpy().astype(np.float32)

    # Evaluate stored actions under the current policy.
    def evaluate_actions(
        self,
        states: CNNObservation,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state_tensors = self._state_to_tensors(states, batched=True)
        logits, values = self.model(state_tensors)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    # Update the policy with one rollout.
    def update(
        self,
        states: CNNObservation,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> dict[str, float]:
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        batch_size = actions.shape[0]
        indices = np.arange(batch_size)

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_indices = indices[start:start + self.config.minibatch_size]
                batch_states: CNNObservation = {
                    key: value[batch_indices]
                    for key, value in states.items()
                }
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]

                new_log_probs, entropy, values = self.evaluate_actions(batch_states, batch_actions)
                log_ratio = new_log_probs - batch_old_log_probs
                ratio = torch.exp(log_ratio)
                unclipped_loss = ratio * batch_advantages
                clipped_loss = torch.clamp(ratio, 1.0 - self.config.clip_range, 1.0 + self.config.clip_range) * batch_advantages
                policy_loss = -torch.min(unclipped_loss, clipped_loss).mean()
                value_loss = F.mse_loss(values, batch_returns)
                entropy_loss = entropy.mean()
                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                approx_kls.append(float(approx_kl.item()))
                self.train_steps += 1

            if approx_kls and approx_kls[-1] > self.config.target_kl:
                break

        return {
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "approx_kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
        }

    # Save the model and training metadata.
    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": asdict(self.config),
                "train_steps": self.train_steps,
            },
            path,
        )

    # Load model weights and optimizer state.
    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.train_steps = int(checkpoint.get("train_steps", 0))


# Load a CNN PPO config dictionary from a checkpoint.
def load_cnn_ppo_config(path: str, device: str = "auto") -> CNNPPOConfig:
    checkpoint = torch.load(path, map_location="cpu")
    config_data = dict(checkpoint["config"])
    config_data.setdefault("frame_stack", 1)
    config_data.setdefault("frame_stack_interval", 1)
    config_data["red_shape"] = tuple(config_data["red_shape"])
    config_data["yellow_shape"] = tuple(config_data["yellow_shape"])
    config_data["blue_shape"] = tuple(config_data["blue_shape"])
    config_data["device"] = device
    return CNNPPOConfig(**config_data)
