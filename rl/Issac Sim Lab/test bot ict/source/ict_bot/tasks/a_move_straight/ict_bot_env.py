# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
import torch
from collections.abc import Sequence

from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG
import isaaclab.sim as sim_utils

# import mdp
import ict_bot.tasks.a_move_straight.mdp as mdp
from isaaclab.envs.mdp import JointVelocityActionCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

##
# Scene definition
##


@configclass
class MoveStraightSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # robots
    robot: ArticulationCfg = ICT_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    wheel_action: JointVelocityActionCfg = JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["right_wheel_joint", "left_wheel_joint"],
        scale=5.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        heading = ObsTerm(
            func=mdp.heading_error_xaxis,
            params={"robot_cfg": SceneEntityCfg("robot")},
        )

        velocity = ObsTerm(
            func=mdp.root_lin_vel_w,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # 1. First: Face the X-axis
    turn_to_x = RewTerm(
        func=mdp.reward_alignment, 
        weight=10.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )
    
    # 2. Second: Move along the X-axis using your 'front'
    move_along_x = RewTerm(
        func=mdp.reward_forward_velocity_along_x, 
        weight=50.0,
        params={"robot_cfg": SceneEntityCfg("robot")},
    )

    alive = RewTerm(
        func=mdp.is_alive, 
        weight=-0.1,
    )


@configclass
class MyEventCfg:
    
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (0.0, 0.0), 
                "y": (0.0, 0.0), 
                "z": (0.1, 0.1),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-3.14, 3.14),  # Random heading (Full 360 degrees)
            },
            "velocity_range": {}, # Sets all velocities to 0
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    out_of_bounds = DoneTerm(func=mdp.out_of_bounds)


##
# Environment configuration
##


@configclass
class MoveStraightEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot to move straight."""

    # Scene settings
    scene: MoveStraightSceneCfg = MoveStraightSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    events: MyEventCfg = MyEventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 10.0
        self.viewer.eye = (2.5, 2.5, 1.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0


##
# Environment class
##


class MoveStraightEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: MoveStraightEnvCfg
    
    def __init__(self, cfg: MoveStraightEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        
        # Find wheel joint indices
        indices, _ = self.scene["robot"].find_joints(self.cfg.wheel_dof_name)
        self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
    
    def _reset_idx(self, env_ids: Sequence[int] | None) -> None:
        """Reset selected environments."""
        super()._reset_idx(env_ids)
        
        # Handle None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        
        num_resets = len(env_ids)
        
        # Reset wheel joint positions and velocities to zero
        num_wheels = len(self._wheel_indices)
        joint_pos = torch.zeros((num_resets, num_wheels), device=self.device)
        joint_vel = torch.zeros((num_resets, num_wheels), device=self.device)
        
        self.scene["robot"].write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._wheel_indices,
            env_ids=env_ids,
        )







'''
Previous code for DirectRLEnvCfg-based environment
'''

# from __future__ import annotations

# import math
# import torch
# from collections.abc import Sequence

# import isaaclab.sim as sim_utils
# from isaaclab.assets import Articulation
# from isaaclab.envs import DirectRLEnv
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

# from .ict_bot_env_cfg import IctBotEnvCfg


# class IctBotEnv(DirectRLEnv):
#     cfg: IctBotEnvCfg

#     def __init__(self, cfg: IctBotEnvCfg, render_mode: str | None = None, **kwargs):
#         super().__init__(cfg, render_mode, **kwargs)
        
#         # Dynamically find the indices based on the names in your config
#         # This returns a list of integers [idx1, idx2]
#         indices, _ = self.robot.find_joints(["right_wheel_joint", "left_wheel_joint"])

#         # 2. Convert to a long tensor for indexing
#         self._wheel_indices = torch.tensor(indices, device=self.device, dtype=torch.long)
        
#         self.R = self.cfg.wheel_radius
#         self.L = self.cfg.wheel_spacing

#         # Store the desired reset pose (position and quaternion)
#         # Position: (0, 0, 0.35) - Z is 0.35 as per robot config
#         # Quaternion: (0.5, 0.5, -0.5, 0.5) - w, x, y, z format
#         self.reset_pos = torch.tensor([0.0, 0.0, 0.35], device=self.device)
#         self.reset_quat = torch.tensor([0.5, 0.5, -0.5, 0.5], device=self.device)

#         # Initialize buffers used in dones/rewards
#         self.y_drift = torch.zeros(self.num_envs, device=self.device)
#         self.forward_vel = torch.zeros(self.num_envs, device=self.device)
#         self.yaw_rate = torch.zeros(self.num_envs, device=self.device)

#     def _setup_scene(self):
#         # create the asset
#         self.robot = Articulation(self.cfg.robot_cfg)
        
#         # add articulation to scene
#         self.scene.articulations["robot"] = self.robot
        
#         # add ground plane
#         spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        
#         # add lights
#         light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
#         light_cfg.func("/World/Light", light_cfg)
        
#         # clone and replicate
#         self.scene.clone_environments(copy_from_source=False)
        
#         # we need to explicitly filter collisions for CPU simulation
#         if self.device == "cpu":
#             self.scene.filter_collisions(global_prim_paths=[])

#     def _pre_physics_step(self, actions: torch.Tensor) -> None:
#         self.actions = actions.clone()

#     def _apply_action(self):
#         actions = self.actions

#         # Scale normalized actions [-1, 1] to max velocities
#         lin_vel = actions[:, 0] * self.cfg.max_linear_velocity
#         ang_vel = actions[:, 1] * self.cfg.max_angular_velocity
        
#         v_left =  (lin_vel - (ang_vel * self.L / 2.0)) / self.R
#         v_right = (lin_vel + (ang_vel * self.L / 2.0)) / self.R
        
#         wheel_speeds = torch.stack([v_left, v_right], dim=-1)
#         self.robot.set_joint_velocity_target(wheel_speeds, joint_ids=self._wheel_indices)


#     def _get_observations(self) -> dict:
#         obs = torch.cat([
#             self.robot.data.root_pos_w,             # [0:3]  (x, y, z)
#             self.robot.data.root_quat_w,            # [3:7]  (w, x, y, z)
#             self.robot.data.root_lin_vel_w,         # [7:10] (vx, vy, vz)
#             self.robot.data.root_ang_vel_w          # [10:13] (wx, wy, wz)
#         ], dim=-1)
#         return {"policy": obs}


#     def _compute_intermediate_values(self):
#         # Get world-space data
#         self.root_pos = self.robot.data.root_pos_w
#         self.root_vel = self.robot.data.root_vel_w
        
#         # Extract values for rewards
#         # Forward velocity (X-axis) = Progress
#         self.forward_vel = self.root_vel[:, 0]
        
#         # Sideways position (Y-axis) = Drift
#         # self.y_drift = self.root_pos[:, 1]
#         # World-frame velocity
#         vel_world = self.robot.data.root_lin_vel_w[:, :2]  # vx, vy

#         # Robot yaw
#         quat = self.robot.data.root_quat_w
#         yaw = torch.atan2(
#             2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
#             1.0 - 2.0 * (quat[:, 2]**2 + quat[:, 3]**2)
#         )

#         # Rotate into body frame
#         sin_yaw = torch.sin(yaw)
#         cos_yaw = torch.cos(yaw)

#         # Body-frame lateral velocity
#         v_lat = -sin_yaw * vel_world[:, 0] + cos_yaw * vel_world[:, 1]

#         self.y_drift = v_lat
        
#         # Angular velocity (Yaw-rate) = Turning
#         self.yaw_rate = self.robot.data.root_ang_vel_w[:, 2]


#     def _get_rewards(self) -> torch.Tensor:
#         # Forward Reward: Reward POSITIVE velocity along local X-axis
#         forward_reward = torch.clamp(self.forward_vel, min=0) * self.cfg.reward_scales["forward_reward"]
        
#         # Backward Penalty: Penalize NEGATIVE velocity along local X-axis
#         backward_penalty = torch.clamp(-self.forward_vel, min=0) * self.cfg.reward_scales["backward_penalty"]
        
#         # Straightness Penalty: Penalize lateral velocity (drifting/sliding)
#         # We use abs() so drifting left or right is equally bad
#         straightness_penalty = torch.abs(self.y_drift) * self.cfg.reward_scales["straightness_penalty"]
        
#         # Heading Penalty: Penalize excessive turning/spinning
#         heading_penalty = torch.abs(self.yaw_rate) * self.cfg.reward_scales["heading_penalty"]
        
#         # Idle Penalty: Penalize if robot is not moving (i.e., forward velocity is low)
#         idle_penalty = (torch.abs(self.forward_vel) < 0.01) * self.cfg.reward_scales["idle_penalty"]
        
#         # Total Reward
#         return forward_reward + backward_penalty + straightness_penalty + heading_penalty + idle_penalty

#     def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
#         # Truncation: Check if the timer (episode_length_buf) reached the limit
#         # max_episode_length is automatically calculated from your episode_length_s
#         time_out = self.episode_length_buf >= self.max_episode_length
        
#         # Termination: Check if the robot drifted too far or turned too much
#         # Drift check (using your y_drift_limit)
#         out_of_bounds = torch.abs(self.y_drift) > self.cfg.y_drift_limit
        
#         # Check if the robot flipped or went out of yaw limits
#         # You'd need to extract 'yaw' in intermediate values for this
#         # died = out_of_bounds | (torch.abs(self.yaw) > self.cfg.yaw_limit)
        
#         died = out_of_bounds
        
#         # Return: (Terminated, Truncated)
#         return died, time_out

#     def _reset_idx(self, env_ids: Sequence[int] | None):
#         # Handle 'None' by converting to a tensor of all environment indices
#         if env_ids is None:
#             env_ids = torch.arange(self.num_envs, device=self.device)
#         else:
#             # Ensure it is a tensor on the correct device (GPU)
#             env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

#         num_resets = len(env_ids)
        
#         # Clone the complete default root state to get height
#         root_states = self.robot.data.default_root_state[env_ids].clone()

        
#         # Set position: add environment spacing, but keep Z from default height
#         root_states[:, 0:2] = self.scene.env_origins[env_ids, 0:2]
        
#         root_states[:, 7:13] = 0  # Reset velocities to zero
        
#         # Write the complete root state to simulation
#         self.robot.write_root_pose_to_sim(root_states[:, 0:7], env_ids=env_ids)
#         self.robot.write_root_velocity_to_sim(root_states[:, 7:13], env_ids=env_ids)

#         self.robot.data.root_pos_w[env_ids] = root_states[:, 0:3]
#         self.robot.data.root_quat_w[env_ids] = root_states[:, 3:7]

#         self.robot.reset(env_ids)

#         # Reset Wheel Joint Positions and Velocities to Zero
#         num_wheels = len(self._wheel_indices)
#         joint_pos = torch.zeros((num_resets, num_wheels), device=self.device)
#         joint_vel = torch.zeros((num_resets, num_wheels), device=self.device)

#         self.robot.write_joint_state_to_sim(
#             joint_pos, 
#             joint_vel, 
#             joint_ids=self._wheel_indices,
#             env_ids=env_ids
#         )

#         # Reset Internal Buffers
#         self.episode_length_buf[env_ids] = 0
#         self.reset_buf[env_ids] = 0