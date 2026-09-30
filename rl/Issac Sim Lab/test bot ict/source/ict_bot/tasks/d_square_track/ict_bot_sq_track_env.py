# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from ict_bot.tasks.c_obstacle_avoidance.ict_bot_obs_env import ObstacleAvoidanceSceneCfg
from ict_bot.tasks.a_move_straight.ict_bot_env import ActionsCfg as MoveStraightActionsCfg
from ict_bot.tasks.a_move_straight.ict_bot_env import MyEventCfg as MoveStraightEventCfg
from ict_bot.tasks.c_obstacle_avoidance.ict_bot_obs_env import TerminationsCfg as ObstacleAvoidanceTerminationsCfg
import isaaclab.sim as sim_utils

import os
from ict_bot import ICT_BOT_ASSETS_DIR


# import mdp
from ict_bot.tasks.c_obstacle_avoidance.mdp.observations import lidar_distances
import ict_bot.tasks.d_square_track.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass


##
# Scene definition
##


@configclass
class SquareTrackEnvSceneCfg(ObstacleAvoidanceSceneCfg):
    """Configuration for the scene."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene is not None:
            self.scene = self.scene.replace(
                spawn=self.scene.spawn.replace(
                    usd_path=os.path.join(ICT_BOT_ASSETS_DIR, "scenes", "square_track.usd")
                )
            )



##
# MDP settings
##


@configclass
class ActionsCfg(MoveStraightActionsCfg):
    """Action specifications for the MDP."""



@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Velocities & Actions
        base_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        actions = ObsTerm(func=mdp.last_action)

        # Obstacle Awareness: 300 rays of depth data
        lidar = ObsTerm(
            func=lidar_distances, 
            params={"sensor_cfg": SceneEntityCfg("raycaster"), "max_distance": 4.0}
        )
        

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # --- POSITIVE MOTIVATION ---
    progress = RewTerm(
        func=mdp.reward_clear_path,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("raycaster"),
            "robot_cfg": SceneEntityCfg("robot"),
        }
    )

    # --- NEGATIVE CONSTRAINTS ---
    no_reverse = RewTerm(
        func=mdp.penalty_anti_reverse, 
        weight=-1000.0, 
        params={"robot_cfg": SceneEntityCfg("robot")}
    )

    action_rate = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.1,
    )

    alive = RewTerm(
        func=mdp.is_alive, 
        weight=-10.0
    )


@configclass
class MyEventCfg(MoveStraightEventCfg):
    """Event specifications for the MDP."""



@configclass
class TerminationsCfg(ObstacleAvoidanceTerminationsCfg):
    """Termination terms for the MDP."""

    target_reached = None



##
# Environment configuration
##


@configclass
class SquareTrackEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for ict bot to move straight."""

    # Scene settings
    scene: SquareTrackEnvSceneCfg = SquareTrackEnvSceneCfg(num_envs=4096, env_spacing=10.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: MyEventCfg = MyEventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.sim.render_interval = self.decimation
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 1.0 / 60.0


##
# Environment class
##


class SquareTrackEnv(ManagerBasedRLEnv):
    """Environment for ICT Bot moving straight."""
    
    cfg: SquareTrackEnvCfg
    
    def __init__(self, cfg: SquareTrackEnvCfg, render_mode: str | None = None, **kwargs):

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