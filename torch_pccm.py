"""Optional PyTorch implementation of the reference PCCM sampler.

This keeps the observation format unchanged. The CUDA path is opt-in because
its speed depends on bullet count, process count, and the host GPU.
"""

from __future__ import annotations

import numpy as np


def pccm_sample_components_torch(
    bullets,
    player_radius: float,
    window: tuple[int, int, int, int],
    sample_shape: tuple[int, int],
    field_w: int,
    field_h: int,
    prediction_frames: int,
    halo_width: float,
    wall_margin: float,
    fps: float = 60.0,
    upper_field_threshold: float = 0.70,
    upper_field_cost: float = 0.30,
    *,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match the NumPy reference and return its four float32 sample grids."""
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("PCCM torch_cuda requested, but PyTorch cannot access a CUDA GPU.")

    from observation_builder import grid_cell_centers

    xx_np, yy_np = grid_cell_centers(window, sample_shape)
    coordinates = torch.as_tensor(np.stack((xx_np, yy_np)), device=device)
    xx, yy = coordinates.unbind(dim=0)
    inside = (xx >= 0.0) & (xx < field_w) & (yy >= 0.0) & (yy < field_h)
    inside_float = inside.to(torch.float32)
    current_cost = torch.zeros(sample_shape, dtype=torch.float32, device=device)
    prediction_cost = torch.zeros_like(current_cost)
    hard_collision = torch.zeros_like(current_cost)

    if bullets:
        x1, y1, x2, y2 = window
        horizon_seconds = prediction_frames / fps
        relevant = []
        for bullet in bullets:
            future_x = bullet.x + bullet.vx * horizon_seconds
            future_y = bullet.y + bullet.vy * horizon_seconds
            padding = bullet.radius + player_radius + halo_width
            if (
                max(bullet.x, future_x) + padding >= x1
                and min(bullet.x, future_x) - padding < x2
                and max(bullet.y, future_y) + padding >= y1
                and min(bullet.y, future_y) - padding < y2
            ):
                relevant.append((bullet.x, bullet.y, bullet.radius, bullet.vx, bullet.vy))

        if relevant:
            packed = np.asarray(relevant, dtype=np.float32)
            state = torch.as_tensor(packed, device=device)
            bx, by, br, vx, vy = state.unbind(dim=1)
            radii = torch.clamp(br + np.float32(player_radius), min=1.0)
            times = torch.arange(prediction_frames + 1, dtype=torch.float32, device=device) / fps
            future_x = bx[:, None] + vx[:, None] * times[None, :]
            future_y = by[:, None] + vy[:, None] * times[None, :]
            dx = xx[None, None] - future_x[:, :, None, None]
            dy = yy[None, None] - future_y[:, :, None, None]
            distances = torch.sqrt(dx * dx + dy * dy)
            falloff = torch.clamp(
                1.0 - torch.clamp(distances - radii[:, None, None, None], min=0.0) / halo_width,
                0.0,
                1.0,
            )
            time_weights = 1.0 - torch.arange(
                prediction_frames + 1, dtype=torch.float32, device=device
            ) / (prediction_frames + 1.0)
            contributions = 0.5 * falloff * time_weights[None, :, None, None]
            contributions = contributions * inside_float[None, None]
            current_cost = 1.0 - torch.prod(1.0 - contributions[:, 0], dim=0)
            predicted = (1.0 - contributions[:, 1:]).reshape(-1, *sample_shape)
            prediction_cost = 1.0 - torch.prod(predicted, dim=0)
            hard_collision = torch.any(
                distances[:, 0] <= radii[:, None, None], dim=0
            ).to(torch.float32)

    horizontal_margin = max(1.0, field_w * wall_margin)
    vertical_margin = max(1.0, field_h * wall_margin)
    wall_cost = torch.zeros_like(current_cost)
    for distance, margin in (
        (xx, horizontal_margin),
        (field_w - xx, horizontal_margin),
        (yy, vertical_margin),
        (field_h - yy, vertical_margin),
    ):
        contribution = 0.5 * torch.clamp(1.0 - distance / margin, 0.0, 1.0) * inside_float
        wall_cost = 1.0 - (1.0 - wall_cost) * (1.0 - contribution)

    upper_boundary = max(1.0, field_h * upper_field_threshold)
    upper_contribution = (
        upper_field_cost * torch.clamp(1.0 - yy / upper_boundary, 0.0, 1.0) * inside_float
    )
    wall_cost = 1.0 - (1.0 - wall_cost) * (1.0 - upper_contribution)
    hard_collision = hard_collision * inside_float

    output = torch.stack((current_cost, prediction_cost, wall_cost, hard_collision)).cpu().numpy()
    return tuple(output[index] for index in range(4))
