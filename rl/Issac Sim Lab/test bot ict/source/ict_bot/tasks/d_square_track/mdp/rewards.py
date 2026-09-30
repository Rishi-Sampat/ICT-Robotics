from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

import torch


def reward_clear_path(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg, 
    robot_cfg: SceneEntityCfg,
    max_lidar_dist: float = 5.0,
    centering_tolerance: float = 0.075 # 7.5cm tolerance
):
    distances = lidar_distances(env, sensor_cfg, max_distance=max_lidar_dist)
    num_rays = distances.shape[1]
    
    # Indices for -Y front (270 deg)
    right_idx = 0
    left_idx = int(num_rays * 0.5)
    front_idx = int(num_rays * 0.75)
    
    # 1. ACTUAL DISTANCES
    dist_l = distances[:, left_idx] * max_lidar_dist
    dist_r = distances[:, right_idx] * max_lidar_dist
    side_diff = torch.abs(dist_l - dist_r)
    
    # 2. THE CENTERING GRADIENT
    # Using 0.1 as the denominator makes the reward drop significantly 
    # if the robot moves more than 10cm from the center.
    centering_reward = torch.exp(-side_diff / 0.1)

    # 3. VELOCITY & ALIGNMENT
    robot = env.scene[robot_cfg.name]
    world_lin_vel = robot.data.root_link_vel_w[:, :3]
    root_quat_w = robot.data.root_link_quat_w
    body_lin_vel = quat_apply(quat_inv(root_quat_w), world_lin_vel)
    forward_vel = -body_lin_vel[:, 1]

    max_indices = torch.argmax(distances, dim=-1)
    # How well is front (270) aligned with the longest path?
    alignment = torch.exp(-torch.abs(max_indices - front_idx).float() / (num_rays * 0.05))

    # 4. GATED SPEED REWARD
    # We only reward speed if the robot is within 'centering_tolerance'
    is_centered = (side_diff < centering_tolerance).float()
    
    # We scale the speed reward by alignment. 
    # This forces the robot to rotate at the square corners.
    reward = 1.0 * centering_reward 
    reward += 5.0 * torch.clamp(forward_vel, min=0.0) * is_centered * alignment

    # 5. THE "WALL-SLIDE" KILLER & CORNERING
    # 90cm track - 15cm robot = 75cm total clearance. 
    # Perfect center = 0.375m on each side.
    # If side distance < 0.2m, it's hugging the wall.
    # If front distance < 0.4m, it's about to hit a corner.
    front_dist = distances[:, front_idx] * max_lidar_dist
    
    collision_risks = (dist_l < 0.2) | (dist_r < 0.2) | (front_dist < 0.4)
    reward[collision_risks] -= 5.0 # Strong negative signal
    
    return reward


def penalty_anti_reverse(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    # In -Y front setup, +Y body velocity is backward.
    backward_vel = robot.data.root_lin_vel_b[:, 1]
    return torch.clamp(backward_vel, min=0.0)
