# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG
from ict_bot.tasks.a_move_straight.ict_bot_env import MoveStraightEnvCfg

##
# Scene definition
##


@configclass
class IctBotEnvCfg(MoveStraightEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 13  # [pos(3), quat(4), lin_vel(3), ang_vel(3)]
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.16
    max_linear_velocity = 0.5
    max_angular_velocity = 2.0
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]
    
    # Reset conditions
    y_drift_limit = 1.0
    yaw_limit = 0.785


@configclass
class IctBotEnvCfg_PLAY(IctBotEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False





'''
Previous code for DirectRLEnvCfg-based environment
'''


# from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG

# from isaaclab.assets import ArticulationCfg
# from isaaclab.envs import DirectRLEnvCfg
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sim import SimulationCfg
# from isaaclab.utils import configclass


# @configclass
# class IctBotEnvCfg(DirectRLEnvCfg):
#     # env
#     decimation = 4
#     episode_length_s = 10.0
#     # - spaces definition
#     action_space = 2        # [linear_vel, angular_vel]
#     observation_space = 13  # Matches the 'obs' tensor we build
#     state_space = 0          # Usually 0 unless using asymmetric actor-critic

#     # simulation
#     sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)

#     # robot(s)
#     robot_cfg: ArticulationCfg = ICT_BOT_CFG.replace(prim_path="/World/envs/env_.*/Robot")

#     # scene
#     num_envs=4096
#     env_spacing=5.0
#     scene: InteractiveSceneCfg = InteractiveSceneCfg(
#         num_envs=num_envs, 
#         env_spacing=env_spacing, 
#         replicate_physics=True
#     )

#     # Physical properties
#     wheel_radius = 0.1
#     wheel_spacing = 0.16
#     max_linear_velocity = 0.5   # meters per second
#     max_angular_velocity = 2.0  # radians per second

#     # custom parameters/scales
#     # - controllable joint
#     wheel_dof_name = [
#         "left_wheel_joint",
#         "right_wheel_joint",
#     ]
#     # - action scale
#     action_scale = 5.0  # [N]
#     # - reward scales
#     reward_scales = {
#         # Reward only POSITIVE velocity along local X-axis
#         "forward_reward": 2.0,        
#         # Penalty specifically for NEGATIVE velocity along local X-axis
#         "backward_penalty": -5.0,     
#         # Penalty for Y-velocity (drifting/sliding)
#         "straightness_penalty": -0.1,  
#         # Penalty for unnecessary spinning
#         "heading_penalty": -0.05,      
#         # Penalty for not moving
#         "idle_penalty": -0.5,
#     }
#     # - reset states/conditions
#     y_drift_limit = 1.0       # Reset if robot drifts > 1m from center
#     yaw_limit = 0.785         # Reset if robot turns > 45 degrees (pi/4)