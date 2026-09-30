from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import heading_error_xaxis
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def reward_alignment(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Reward for facing World X. Max when error is 0."""
    error = heading_error_xaxis(env, robot_cfg).squeeze(-1)
    return torch.exp(-torch.abs(error) / 0.25)


def reward_forward_velocity_along_x(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Reward for moving 'Forward' (-Y) in the 'World X' direction."""
    robot = env.scene[robot_cfg.name]
    
    # 1. Get world velocity
    vel_w = robot.data.root_lin_vel_w[:, 0:3]
    
    # 2. Get the robot's local 'Forward' vector (-Y) in World Frame
    # We rotate [0, -1, 0] by the robot's orientation
    local_front = torch.tensor([0.0, -1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    forward_w = quat_apply(robot.data.root_quat_w, local_front)
    
    # 3. Dot product: How much of our velocity is going 'Forward' AND along 'World X'?
    # Since World X is [1, 0, 0], this is: (Forward_World_X_Component) * (Velocity_World_X_Component)
    dot_progress = forward_w[:, 0] * vel_w[:, 0]
    
    return torch.clamp(dot_progress, min=0.0)

