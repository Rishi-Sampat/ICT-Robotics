from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg


def filtered_illegal_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the contact force ONLY from filtered objects exceeds the threshold."""
    # 1. Extract the contact sensor
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # 2. Grab the filtered history matrix 
    # Shape: (N_envs, T_history, B_bodies, M_filtered_objects, 3_forces)
    filtered_forces = contact_sensor.data.force_matrix_w_history
    
    # 3. If nothing has loaded yet or matrix is empty, return no termination
    if filtered_forces is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        
    # 4. Filter down to the body IDs specified in your SceneEntityCfg (wheels/base)
    # slice the history, selected bodies, and all filtered obstacles
    selected_body_forces = filtered_forces[:, :, sensor_cfg.body_ids, :, :]
    
    # 5. Compute the force magnitudes (norms) across the X, Y, Z vector components
    force_magnitudes = torch.norm(selected_body_forces, dim=-1) # Shape: (N, T, B, M)
    
    # 6. Find the maximum force across history (T), sensor bodies (B), and filtered objects (M)
    max_forces_per_env = torch.max(
        torch.max(
            torch.max(force_magnitudes, dim=1)[0], # max across history
            dim=1)[0], # max across sensor bodies
        dim=1)[0] # max across filtered obstacles
    
    # 7. Reset the env if the highest force from a filtered object exceeds your threshold
    return max_forces_per_env > threshold