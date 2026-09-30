from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.b_reach_target.mdp.observations import heading_error
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def heading_error_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Reward: High when -Y face is aligned with target. Returns flat [N] tensor."""
    # 1. Call the observation function: returns [4096, 1]
    error_obs = heading_error(env, robot_cfg)
    
    # 2. Squeeze the last dimension: returns [4096]
    error_rad = error_obs.squeeze(-1)
    
    # 3. Apply Cosine: 1.0 (facing) to -1.0 (away)
    return torch.cos(error_rad)


def reward_gated_progress_neg_y(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Rewards moving closer ONLY if the -Y face is pointed at the target."""
    # 1. Distance Delta
    current_dist = torch.norm(env.target_pos - env.scene[robot_cfg.name].data.root_pos_w, dim=-1)
    if not hasattr(env, "prev_tgt_dist"):
        env.prev_tgt_dist = current_dist.clone()
    dist_delta = env.prev_tgt_dist - current_dist
    env.prev_tgt_dist = current_dist.clone()

    # 2. Strict Alignment (Must be within ~11 degrees)
    error_rad = heading_error(env, robot_cfg).squeeze(-1)
    alignment = torch.cos(error_rad)
    
    # 3. THE GATE: 0.98 is a very tight window.
    # If the robot isn't pointing its 'face' almost exactly at the cone, 
    # it gets ZERO progress reward.
    gate = (alignment > 0.95).float()
    
    # Also, we penalize moving closer if NOT aligned (The 'Anti-Spiral' rule)
    return torch.where(gate > 0, dist_delta * 5.0, -torch.abs(dist_delta) * 2.0)


def penalize_backwards_movement_neg_y(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """Penalizes velocity in the +Y direction (backwards for this robot)."""
    robot = env.scene[robot_cfg.name]
    
    # 1. Get world velocity: (num_envs, 3)
    vel_w = robot.data.root_lin_vel_w
    
    # 2. Transform to local frame
    inv_quat = quat_inv(robot.data.root_quat_w)
    vel_local = quat_apply(inv_quat, vel_w)
    
    # 3. Extract Y-velocity (Local +Y is backwards)
    # If vel_y > 0, the robot is reversing.
    vel_y = vel_local[:, 1]
    
    # 4. Return penalty: only active if moving backwards
    # We use .clamp(min=0) so moving forward (negative Y) gives 0.0 reward
    return torch.clamp(vel_y, min=0.0)

