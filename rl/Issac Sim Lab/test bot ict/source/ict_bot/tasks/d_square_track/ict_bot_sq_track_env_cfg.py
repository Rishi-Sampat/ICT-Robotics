# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import mdp
from isaaclab.utils import configclass
from ict_bot.tasks.d_square_track.ict_bot_sq_track_env import SquareTrackEnvCfg

##
# Scene definition
##


@configclass
class IctBotSqTrackEnvCfg(SquareTrackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

    # Action/Observation/State spaces
    action_space = 2        # [linear_vel, angular_vel]
    observation_space = 316  # 316 = 3 (root pos) + 4 (root quat) + 2*2 (wheel joint pos/vel) + 300 (raycast)
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
class IctBotSqTrackEnvCfg_PLAY(IctBotSqTrackEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.episode_length_s = 1000000000.0
        # make a smaller scene for play
        self.scene.num_envs = 1
        # disable randomization for play
        self.observations.policy.enable_corruption = False