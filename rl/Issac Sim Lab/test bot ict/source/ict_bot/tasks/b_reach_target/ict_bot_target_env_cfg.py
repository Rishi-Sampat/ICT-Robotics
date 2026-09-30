# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot.assets.robots.ict_bot import ICT_BOT_CFG
from ict_bot.tasks.b_reach_target.ict_bot_target_env import ReachTargetEnvCfg


##
# Scene definition
##


@configclass
class IctBotTargetEnvCfg(ReachTargetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 13  # [Relative Target Pos (3), Heading Error (1), Base Linear Vel (2), Base Angular Vel (2), Last Action (2), Lidar Distances (300)]
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.16
    max_linear_velocity = 0.6
    max_angular_velocity = 3.5
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]

    yaw_limit = 6.28  # allow max 360 degrees turns


@configclass
class IctBotTargetEnvCfg_PLAY(IctBotTargetEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False