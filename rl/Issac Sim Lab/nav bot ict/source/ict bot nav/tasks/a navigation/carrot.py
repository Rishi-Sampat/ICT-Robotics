from __future__ import annotations
import torch

# ---------------------------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------------------------
MARKER_HEIGHT   = 0.25
ADVANCE_OFFSET  = 0.25   # Distance before the corner to trigger the next carrot
LINE_HALF_WIDTH = 1.2    # Lateral tolerance for crossing the line

# ---------------------------------------------------------------------------
# Hardcoded Geometry Maps (Local Corridor Frame)
# ---------------------------------------------------------------------------
# Path A: Spawns at top left, moves +X, Turns Right, moves -Y
WAYPOINTS_A = [
    # Segment 1: Moving +X
    {'target': [-1.5, 4.5], 'trigger': [-1.75, 4.5], 'tangent': [ 1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [ 0.0, 4.5], 'trigger': [-0.25, 4.5], 'tangent': [ 1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [ 1.5, 4.5], 'trigger': [ 1.25, 4.5], 'tangent': [ 1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [ 3.0, 4.5], 'trigger': [ 2.75, 4.5], 'tangent': [ 1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    
    # Corner: 45 Degree Miter
    {'target': [ 4.5, 4.5], 'trigger': [ 4.25, 4.5], 'tangent': [ 0.7071, -0.7071], 'lateral': [0.7071, 0.7071]},
    
    # Segment 2: Moving -Y
    {'target': [ 4.5, 3.0], 'trigger': [ 4.5,  3.25], 'tangent': [ 0.0, -1.0], 'lateral': [ 1.0, 0.0]},
    {'target': [ 4.5, 1.5], 'trigger': [ 4.5,  1.75], 'tangent': [ 0.0, -1.0], 'lateral': [ 1.0, 0.0]},
    {'target': [ 4.5, 0.0], 'trigger': [ 4.5,  0.25], 'tangent': [ 0.0, -1.0], 'lateral': [ 1.0, 0.0]},
    {'target': [ 4.5,-1.5], 'trigger': [ 4.5, -1.25], 'tangent': [ 0.0, -1.0], 'lateral': [ 1.0, 0.0]},
]

# Path B: Spawns at bottom right, moves +Y, Turns Left, moves -X
WAYPOINTS_B = [
    # Segment 1: Moving +Y
    {'target': [ 4.5,-1.5], 'trigger': [ 4.5, -1.75], 'tangent': [ 0.0,  1.0], 'lateral': [ 1.0,  0.0]},
    {'target': [ 4.5, 0.0], 'trigger': [ 4.5, -0.25], 'tangent': [ 0.0,  1.0], 'lateral': [ 1.0,  0.0]},
    {'target': [ 4.5, 1.5], 'trigger': [ 4.5,  1.25], 'tangent': [ 0.0,  1.0], 'lateral': [ 1.0,  0.0]},
    {'target': [ 4.5, 3.0], 'trigger': [ 4.5,  2.75], 'tangent': [ 0.0,  1.0], 'lateral': [ 1.0,  0.0]},
    
    # Corner: 45 Degree Miter
    {'target': [ 4.5, 4.5], 'trigger': [ 4.5,  4.25], 'tangent': [-0.7071, 0.7071], 'lateral': [-0.7071, -0.7071]},
    
    # Segment 2: Moving -X
    {'target': [ 3.0, 4.5], 'trigger': [ 3.25,  4.5], 'tangent': [-1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [ 1.5, 4.5], 'trigger': [ 1.75,  4.5], 'tangent': [-1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [ 0.0, 4.5], 'trigger': [ 0.25,  4.5], 'tangent': [-1.0,  0.0], 'lateral': [ 0.0, 1.0]},
    {'target': [-1.5, 4.5], 'trigger': [-1.25,  4.5], 'tangent': [-1.0,  0.0], 'lateral': [ 0.0, 1.0]},
]

# ---------------------------------------------------------------------------
# Initialization Logic
# ---------------------------------------------------------------------------
def _init_global_geometry_tensors(env, cfg):
    """Loads the hardcoded static geometry into global PyTorch tensors."""
    device = cfg.sim.device
    
    # Shape for all lookup tables: [num_paths, num_waypoints, 2]
    # Index 0 = Path A, Index 1 = Path B
    env.static_targets = torch.tensor([[w['target'] for w in WAYPOINTS_A], 
                                       [w['target'] for w in WAYPOINTS_B]], device=device, dtype=torch.float32)
    
    env.static_triggers = torch.tensor([[w['trigger'] for w in WAYPOINTS_A], 
                                        [w['trigger'] for w in WAYPOINTS_B]], device=device, dtype=torch.float32)
                                        
    env.static_tangents = torch.tensor([[w['tangent'] for w in WAYPOINTS_A], 
                                        [w['tangent'] for w in WAYPOINTS_B]], device=device, dtype=torch.float32)
                                        
    env.static_laterals = torch.tensor([[w['lateral'] for w in WAYPOINTS_A], 
                                        [w['lateral'] for w in WAYPOINTS_B]], device=device, dtype=torch.float32)


def place_carrot(env, env_ids: torch.Tensor) -> None:
    """Called at episode reset. Assigns start conditions to given envs."""
    if env_ids.dtype == torch.bool:
        idx = env_ids.nonzero(as_tuple=False).squeeze(-1)
    else:
        idx = env_ids
    if len(idx) == 0:
        return

    n = len(idx)
    device = env.device

    # Initialize the static lookup tables exactly once
    if not hasattr(env, "static_targets"):
        _init_global_geometry_tensors(env)

    # Initialize buffers
    if not hasattr(env, "carrot_pass_count"):
        env.carrot_pass_count = torch.zeros(env.num_envs, device=device)
    if not hasattr(env, "waypoint_idx"):
        env.waypoint_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    if not hasattr(env, "spawn_end"):
        env.spawn_end = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    env.carrot_pass_count[idx] = 0.0
    env.waypoint_idx[idx]      = 0

    # Determine spawn end
    robot_pos = env.scene["robot"].data.root_pos_w
    robot_y   = robot_pos[idx, 1]
    env_origin_y = env.scene.env_origins[idx, 1]
    local_y = robot_y - env_origin_y   

    # local_y > 0 means top corridor (Spawn A), else bottom corridor (Spawn B)
    is_end_a = local_y > 0.0
    env.spawn_end[idx] = torch.where(is_end_a,
                                      torch.zeros(n, dtype=torch.long, device=device),
                                      torch.ones(n,  dtype=torch.long, device=device))

    # Set initial waypoint targets
    _set_carrot_from_waypoint(env, idx)

    # Initialize prev_tgt_dist for the reward function
    diff = env.target_pos[idx, :2] - robot_pos[idx, :2]
    env.prev_tgt_dist[idx] = torch.norm(diff, dim=-1)

# ---------------------------------------------------------------------------
# Per-Step Update Logic
# ---------------------------------------------------------------------------
def update_carrot(env) -> None:
    """Called every step to check waypoint progression via vector projection."""
    robot_pos = env.scene["robot"].data.root_pos_w[:, :2]  # [N, 2]
    
    e_idx = env.spawn_end
    w_idx = env.waypoint_idx
    
    # Clamp w_idx to prevent indexing out of bounds on the final waypoint
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(w_idx, max=max_idx)
    
    # Vectorized Lookup: Instantly extract the geometry for the current state [N, 2]
    local_triggers = env.static_triggers[e_idx, w_idx_safe]
    tangents       = env.static_tangents[e_idx, w_idx_safe]
    laterals       = env.static_laterals[e_idx, w_idx_safe]

    # Convert triggers to world coordinates
    world_triggers = env.scene.env_origins[:, :2] + local_triggers

    # Calculate vector from the trigger line to the robot
    vec_to_robot = robot_pos - world_triggers
    
    # Calculate projections
    forward_dot = (vec_to_robot * tangents).sum(dim=-1)
    lateral_dot = (vec_to_robot * laterals).sum(dim=-1)

    # Check conditions: Has crossed the line AND is within path width
    crossed_line = forward_dot > 0.0
    within_width = torch.abs(lateral_dot) < LINE_HALF_WIDTH
    reached = crossed_line & within_width

    # Advance logic
    max_waypoint = torch.full_like(env.waypoint_idx, max_idx)
    not_at_end = env.waypoint_idx < max_waypoint
    advancing  = reached & not_at_end

    env.carrot_pass_count += advancing.float()
    env.waypoint_idx = torch.where(advancing,
                                    env.waypoint_idx + 1,
                                    env.waypoint_idx)

    # Update world targets strictly for environments that advanced
    if advancing.any():
        adv_ids = advancing.nonzero(as_tuple=False).squeeze(-1)
        _set_carrot_from_waypoint(env, adv_ids)


def _set_carrot_from_waypoint(env, env_ids: torch.Tensor) -> None:
    """Vectorized assignment of the XYZ target coordinate."""
    e_idx = env.spawn_end[env_ids]
    w_idx = env.waypoint_idx[env_ids]
    
    max_idx = env.static_targets.shape[1] - 1
    w_idx_safe = torch.clamp(w_idx, max=max_idx)
    
    # Lookup the [X, Y] target for the specific advancing environments
    local_targets = env.static_targets[e_idx, w_idx_safe]
    
    # Assign WORLD coordinates (Local + Env Origin)
    env.target_pos[env_ids, :2] = env.scene.env_origins[env_ids, :2] + local_targets
    env.target_pos[env_ids, 2]  = env.scene.env_origins[env_ids, 2] + MARKER_HEIGHT
