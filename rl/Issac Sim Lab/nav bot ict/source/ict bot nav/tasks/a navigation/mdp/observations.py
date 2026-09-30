from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.utils.math import quat_inv, quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def rel_line_dist(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """
    Returns [forward_dist_to_line, lateral_offset_from_centreline].
    Uses the absolute waypoint geometry and lateral triggers.
    """
    robot = env.scene[robot_cfg.name]

    # Fetch geometry mappings
    e_idx = env.spawn_end
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(env.waypoint_idx, max=max_idx)
    
    corridor_fwd = env.static_tangents[e_idx, w_idx_safe]  # [N, 2]
    lateral_vec  = env.static_laterals[e_idx, w_idx_safe]  # [N, 2]
    local_triggers = env.static_triggers[e_idx, w_idx_safe]

    robot_pos = robot.data.root_pos_w[:, :2]
    target_pos = env.target_pos[:, :2]

    # 1. Forward Distance (projected onto tangent)
    to_carrot = target_pos - robot_pos
    forward_dist = (to_carrot * corridor_fwd).sum(dim=-1, keepdim=True)

    # 2. Lateral Offset (projected onto lateral vector from the trigger line)
    world_triggers = env.scene.env_origins[:, :2] + local_triggers
    vec_to_robot = robot_pos - world_triggers
    
    # Divided by 1.0 (LINE_HALF_WIDTH) to normalize
    lateral_offset = (vec_to_robot * lateral_vec).sum(dim=-1, keepdim=True) / 1.0

    return torch.cat([forward_dist, lateral_offset], dim=-1)  # [N, 2]



def heading_to_line(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg):
    """
    Angle from robot forward to the absolute corridor forward (tangent).
    Shape: [N, 2] — [sin, cos] of angle
    """
    robot = env.scene[robot_cfg.name]

    # Fetch exact corridor forward
    e_idx = env.spawn_end
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(env.waypoint_idx, max=max_idx)
    
    corridor_fwd = env.static_tangents[e_idx, w_idx_safe]  # [N, 2]

    # Pad to 3D to apply quaternion rotation
    pos_w = torch.cat([
        corridor_fwd, 
        torch.zeros(env.num_envs, 1, device=env.device)
    ], dim=-1)

    # Rotate into robot local frame
    q_inv = quat_inv(robot.data.root_quat_w)
    local_pos = quat_apply(q_inv, pos_w)[:, :2]  # [N, 2]

    # Angle in robot frame — forward is local -Y
    angle = torch.atan2(local_pos[:, 0], -local_pos[:, 1])
    
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)



def joint_velocity(env, robot_cfg: SceneEntityCfg):
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_vel[:, robot_cfg.joint_ids]  # [N, 2] — raw rad/s

def root_lin_vel_b_2d(env, robot_cfg):
    return env.scene[robot_cfg.name].data.root_lin_vel_b[:, :2]  # [vx, vy]

def root_ang_vel_b_z(env, robot_cfg):
    return env.scene[robot_cfg.name].data.root_ang_vel_b[:, 2:3]  # [wz]



def lidar_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    num_beams: int = 360,
):
    if sensor_cfg.name not in env.scene.keys():
        return torch.zeros(env.num_envs, num_beams, device=env.device)

    sensor    = env.scene[sensor_cfg.name]
    robot_pos = env.scene["robot"].data.root_pos_w.unsqueeze(1)
    hits      = sensor.data.ray_hits_w
    distances = torch.norm(hits - robot_pos, dim=-1)

    max_range = 4.0
    distances = torch.clamp(distances, 0.0, max_range) / max_range

    # 1. Gaussian measurement noise — already present, keep as-is
    noise = torch.randn_like(distances) * 0.02
    lidar_t = torch.clamp(distances + noise, 0.0, 1.0)

    # 2. Beam dropout — reflective surfaces, dust, out-of-range
    # Real RPLidar A-series drops 1–3% of beams on glass/dark surfaces
    dropout_mask = torch.rand_like(lidar_t) < 0.02
    lidar_t = torch.where(dropout_mask, torch.ones_like(lidar_t), lidar_t)

    # 3. Random short-range false returns — spurious reflections
    # Simulates multipath returns from shiny floors at 0.1–0.3m
    false_return_mask = torch.rand_like(lidar_t) < 0.005
    false_dist = torch.rand_like(lidar_t) * 0.1   # [0, 0.1] normalised = [0, 0.4m]
    lidar_t = torch.where(false_return_mask, false_dist, lidar_t)

    # 4. Per-env scan offset — simulates LiDAR mounting angle error ±3°
    # Implemented as a circular shift of beams per env, randomised at reset
    if not hasattr(env, "_lidar_beam_offset") or env._lidar_beam_offset is None:
        env._lidar_beam_offset = torch.randint(
            -3, 4, (env.num_envs,), device=env.device
        )
    offset = env._lidar_beam_offset   # [N], integer beam shift
    # Vectorised circular roll per env
    idx = (torch.arange(num_beams, device=env.device).unsqueeze(0)
            - offset.unsqueeze(1)) % num_beams        # [N, num_beams]
    lidar_t = lidar_t.gather(1, idx)

    return lidar_t



