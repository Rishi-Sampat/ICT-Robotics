# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the ICT Bot."""

import os
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# Path to the directory containing this Python script
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# Construct the relative path to the USD file
USD_PATH = os.path.join(ROOT_DIR, "ict_bot.usd")

ICT_BOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1.0,
            max_angular_velocity=1.5,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            "right_wheel_joint": 0.0,
            "left_wheel_joint": 0.0,
        },
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["right_wheel_joint", "left_wheel_joint"],
            effort_limit_sim=3.0,
            velocity_limit_sim=15.0,
            stiffness=0.0,
            damping=1.0,
        ),
    }
)