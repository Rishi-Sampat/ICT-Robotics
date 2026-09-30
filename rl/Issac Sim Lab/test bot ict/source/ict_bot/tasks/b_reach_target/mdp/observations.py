from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def rel_target_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Target position in robot's USD local frame."""
    robot = env.scene[robot_cfg.name]
    target = env.target_pos
    pos_w = target - robot.data.root_pos_w
    q_inv = quat_inv(robot.data.root_quat_w)
    return quat_apply(q_inv, pos_w)


def heading_error(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Angle to target where 0.0 rad is the -Y axis (The Face)."""
    local_pos = rel_target_pos(env, robot_cfg)
    # Standard atan2(y, x) is for +X. For -Y front, we use:
    # atan2(Side_X, Forward_-Y)
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    return angle.unsqueeze(-1)


def target_reached(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, distance: float) -> torch.Tensor:
    """Checks if the robot is within a specific radius of the target."""
    robot = env.scene[robot_cfg.name]
    # target = env.scene[target_cfg.name]
    target = env.target_pos
    
    # Calculate Euclidean distance (L2 norm) in world frame
    # root_pos_w is (num_envs, 3)
    tgt_dist = torch.norm(target - robot.data.root_pos_w, dim=-1)
    
    # Return a boolean tensor (Shape: [num_envs])
    # IMPORTANT: Isaac Lab Terminations expect Bool, Rewards cast this to Float automatically
    return tgt_dist <= distance