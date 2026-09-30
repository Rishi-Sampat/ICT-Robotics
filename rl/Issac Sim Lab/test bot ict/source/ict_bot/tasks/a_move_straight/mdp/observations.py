from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_apply, quat_inv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.scene import SceneEntityCfg


def heading_error_xaxis(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    q_inv = quat_inv(robot.data.root_quat_w)
    
    # Target direction: World X [1, 0, 0]
    world_x = torch.zeros_like(robot.data.root_pos_w)
    world_x[:, 0] = 1.0
    
    # Project World X into Robot Local Frame
    local_xaxis = quat_apply(q_inv, world_x)
    
    # FRONT = -Y axis, SIDE = +X axis
    # atan2(side, forward) -> atan2(local_xaxis.x, -local_xaxis.y)
    return torch.atan2(local_xaxis[:, 0], -local_xaxis[:, 1]).unsqueeze(-1)


