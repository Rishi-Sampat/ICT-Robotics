from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_target_marker_in_ring(env: ManagerBasedRLEnv, env_ids: torch.Tensor, radius_range: tuple[float, float], z_height: float):
    num_resets = len(env_ids)
    device = env.device
    
    # Generate random local polar coordinates
    r = torch.empty(num_resets, device=device).uniform_(*radius_range)
    theta = torch.empty(num_resets, device=device).uniform_(0, 2 * torch.pi)
    
    # Convert to local XYZ
    x_local = r * torch.cos(theta)
    y_local = r * torch.sin(theta)
    
    # ADD ENV ORIGINS to get World Coordinates
    env.target_pos[env_ids, 0] = x_local + env.scene.env_origins[env_ids, 0]
    env.target_pos[env_ids, 1] = y_local + env.scene.env_origins[env_ids, 1]
    env.target_pos[env_ids, 2] = z_height + env.scene.env_origins[env_ids, 2]