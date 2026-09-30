from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances, heading_error

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

import torch
from isaaclab.utils.math import quat_apply



def reward_robust_navigation(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    target = env.target_pos
    
    # --- 1. COORDINATE SETUP ---
    quat = robot.data.root_quat_w
    front_body = torch.tensor([0.0, -1.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    front_world = quat_apply(quat, front_body)[:, :2]
    front_world = front_world / (torch.norm(front_world, dim=-1, keepdim=True) + 1e-6)

    target_vec = target[:, :2] - robot.data.root_pos_w[:, :2]
    target_dist = torch.norm(target_vec, dim=-1, keepdim=True).squeeze(-1)
    target_dir = target_vec / (target_dist.unsqueeze(-1) + 1e-6)

    # --- 2. MULTI-ZONE LIDAR ---
    norm_dist = lidar_distances(env, sensor_cfg, max_distance=3.0) 
    num_rays = norm_dist.shape[1]
    face_idx = int(num_rays * 0.75)
    
    front_shield = norm_dist[:, (face_idx-15):(face_idx+15)]
    min_clearance = torch.min(front_shield, dim=-1).values 
    
    # --- 3. DYNAMIC LOGIC SWITCH (The Final Approach) ---
    # If target is closer than 1.0m, we reduce obstacle avoidance to 0
    # to prevent the robot from "dodging" the target or nearby walls.
    is_near_target = target_dist < 1.0
    obstacle_weight = torch.where(is_near_target, 0.0, 1.0)

    # --- 4. TARGET ALIGNMENT & PROGRESS ---
    dot_product = torch.sum(front_world * target_dir, dim=-1)
    
    # Sharp alignment for the finish line
    alignment_bonus = torch.clamp(dot_product, min=0.0) ** 4 * 20.0
    
    # Distance Reduction
    progress = torch.sum(robot.data.root_lin_vel_w[:, :2] * target_dir, dim=-1)
    # Give a massive boost to progress when near the target
    progress_scale = torch.where(is_near_target, 80.0, 40.0)
    reward_progress = torch.clamp(progress, min=0.0) * progress_scale

    # --- 5. OBSTACLE ESCAPE (Disabled when near target) ---
    # Use peripheral 180-deg sweep
    peripheral_zone = norm_dist[:, (face_idx - num_rays//4):(face_idx + num_rays//4)]
    max_peripheral_clearance = torch.max(peripheral_zone, dim=-1).values
    is_blocked = min_clearance < 0.166 # 50cm
    
    reward_escape = torch.where(is_blocked, max_peripheral_clearance * 25.0, 0.0) * obstacle_weight

    # --- 6. SUCCESS GRAVITY ---
    # A localized "Vacuum" reward for the final 50cm
    gravity = torch.where(target_dist < 0.5, 10.0 / (target_dist + 0.05), 0.0)

    # --- 7. FORWARD SPEED ---
    forward_speed = torch.clamp(-robot.data.root_lin_vel_b[:, 1], min=0.0)
    reward_speed = forward_speed * 5.0

    return reward_progress + alignment_bonus + reward_escape + gravity + reward_speed


def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    # In -Y front setup, +Y body velocity is backward.
    backward_vel = robot.data.root_lin_vel_b[:, 1]
    return torch.clamp(backward_vel, min=0.0)
