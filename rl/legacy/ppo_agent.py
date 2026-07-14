from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F


@dataclass
class PPOConfig:
    state_dim: int
    action_dim: int = 9
    hidden_dim_1: int = 64
    hidden_dim_2: int = 32
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


class ActorCritic(nn.Module):
    # Create a shared actor-critic network.
    def __init__(self, state_dim: int, action_dim: int, hidden_dim_1: int, hidden_dim_2: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim_1),
            nn.Tanh(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim_2, action_dim)
        self.critic = nn.Linear(hidden_dim_2, 1)

    # Return action logits and state value.
    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(states)
        logits = self.actor(features)
        values = self.critic(features).squeeze(-1)
        return logits, values


class PPOAgent:
    # Create the PPO model and optimizer.
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = self._select_device(config.device)
        self.model = ActorCritic(config.state_dim, config.action_dim, config.hidden_dim_1, config.hidden_dim_2).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.train_steps = 0

    # Choose CPU or CUDA for training.
    def _select_device(self, requested_device: str) -> torch.device:
        if requested_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(requested_device)

    # Sample one action from the current policy.
    def select_action(self, state: np.ndarray) -> tuple[int, float, float]:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, values = self.model(state_tensor)
            distribution = Categorical(logits=logits)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
        return int(action.item()), float(log_prob.item()), float(values.item())

    # Choose the highest-probability action for evaluation.
    def select_greedy_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.model(state_tensor)
        return int(torch.argmax(logits, dim=1).item())

    # Return action probabilities for debugging.
    def action_probs(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits, _ = self.model(state_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
        return probs.detach().cpu().numpy()

    # Return the critic value for one state.
    def state_value(self, state: np.ndarray) -> float:
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            _, value = self.model(state_tensor)
        return float(value.item())

    # Evaluate stored actions under the current policy.
    def evaluate_actions(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.model(states)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    # Update the policy with one rollout.
    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        returns: np.ndarray,
        advantages: np.ndarray,
    ) -> dict[str, float]:
        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_tensor = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std(unbiased=False) + 1e-8)

        policy_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        indices = np.arange(states.shape[0])

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                batch_indices = indices[start:start + self.config.minibatch_size]
                batch_states = states_tensor[batch_indices]
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


# Load a PPO config dictionary from a checkpoint.
def load_ppo_config(path: str, device: str = "auto") -> PPOConfig:
    checkpoint = torch.load(path, map_location="cpu")
    config_data = dict(checkpoint["config"])
    config_data["device"] = device
    return PPOConfig(**config_data)
