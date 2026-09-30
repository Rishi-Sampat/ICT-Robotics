from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode terminates if max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length


def out_of_bounds(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode terminates if robot drifts too far laterally (Y-axis in world frame)."""
    vel_y = env.scene["robot"].data.root_lin_vel_w[:, 1]
    return torch.abs(vel_y) > env.cfg.y_drift_limit