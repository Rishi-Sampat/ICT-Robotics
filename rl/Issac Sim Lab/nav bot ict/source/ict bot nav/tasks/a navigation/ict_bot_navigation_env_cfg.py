# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot_nav.tasks.a_navigation.ict_bot_navigation_env import NavigationEnvCfg

##
# Scene definition
##


@configclass
class IctBotNavigationEnvCfg(NavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 369  # [RelDist(2), Head(2), WheelVel(2), RobotVel(2), RobotAngVel(1), Lidar(360)]
    state_space = 0
    
    # Physical properties
    wheel_radius = 0.1
    wheel_spacing = 0.16
    max_linear_velocity = 2.0
    max_angular_velocity = 15.0
    
    # Custom parameters/scales
    wheel_dof_name = [
        "left_wheel_joint",
        "right_wheel_joint",
    ]

    yaw_limit = 6.28  # allow max 360 degrees turns


@configclass
class IctBotNavigationEnvCfg_PLAY(IctBotNavigationEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 1
        self.episode_length_s = 35.0

        # disable randomization for play
        self.observations.policy.enable_corruption = False

        # Lock obstacles at a specific curriculum level for play
        self.curriculum.obstacle_difficulty.params["cfg"].play_level = 8