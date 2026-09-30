from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def check_target_reached(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, threshold: float = 0.3):
    robot = env.scene[robot_cfg.name]

    # Calculate distance ignoring Z
    diff = env.target_pos[:, :2] - robot.data.root_pos_w[:, :2]
    dist = torch.norm(diff, dim=-1)

    return dist < threshold