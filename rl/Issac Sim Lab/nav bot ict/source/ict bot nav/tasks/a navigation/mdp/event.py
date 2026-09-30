from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.envs.mdp import reset_root_state_uniform

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg



def reset_robot_l_corridor(env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg):
    """
    Randomly assigns environments to End A or End B, sets the pose ranges, 
    and triggers a uniform reset, including zeroing out velocities.
    """
    n = len(env_ids)
    
    # 1. Randomly decide End A (0) or End B (1)
    spawn_ends = torch.randint(0, 2, (n,), device=env.device)
    
    # Save the decision so the carrot tracker knows which path to follow
    env.spawn_end[env_ids] = spawn_ends
    
    # 2. Define velocity range to ensure robot starts completely still
    zero_velocity = {
        "linear":  {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        "angular": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    }
    
    # 3. Process each spawn end
    for end_val in [0, 1]:
        mask = (spawn_ends == end_val)
        subset_ids = env_ids[mask]
        
        if len(subset_ids) == 0:
            continue
            
        # Define the specific range for this subset
        # End A: X=[-3.2, -2.8], Y=[4.2, 4.8] 
        # End B: X=[4.2, 4.8],   Y=[-3.2, -2.8]
        current_range = {
            "x": (-3.2, -2.8) if end_val == 0 else (4.2, 4.8),
            "y": (4.2, 4.8) if end_val == 0 else (-3.2, -2.8),
            "z": (0.1, 0.1),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-3.14, 3.14) # Full 360 rotation
        }
        
        # 4. Use the built-in function
        reset_root_state_uniform(
            env=env, 
            env_ids=subset_ids, 
            asset_cfg=asset_cfg, 
            pose_range=current_range,
            velocity_range=zero_velocity
        )


def randomize_wheel_slip(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    slip_range: tuple[float, float] = (0.95, 1.05),
) -> None:
    """
    Apply independent slip scale to left and right wheel damping.
    Simulates asymmetric tyre-floor contact — ±5% per wheel independently.
    """
    asset = env.scene[asset_cfg.name]
    indices, _ = asset.find_joints(["left_wheel_joint", "right_wheel_joint"])
    indices = torch.tensor(indices, device=env.device)

    n = len(env_ids)
    # Independent scale per wheel per env — [n, 2]
    scales = torch.empty(n, 2, device=env.device).uniform_(*slip_range)

    current_damping = asset.data.default_joint_damping[env_ids][:, indices]
    new_damping = current_damping * scales

    asset.write_joint_damping_to_sim(
        new_damping, joint_ids=indices, env_ids=env_ids
    )