from __future__ import annotations

import dataclasses
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def _rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=True,
        kinematic_enabled=True,
        max_linear_velocity=3.0,
        max_angular_velocity=0.0,
        retain_accelerations=False,
    )


def _mass() -> sim_utils.MassPropertiesCfg:
    return sim_utils.MassPropertiesCfg(mass=50.0)


def _collision() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True)


# ---------------------------------------------------------------------------
# Cube variants
# side lengths: large=0.6, medium=0.4, small=0.2 — all 0.7 m tall
# ---------------------------------------------------------------------------

CUBE_XLARGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CubeXLarge",
    spawn=sim_utils.CuboidCfg(
        size=(0.7, 0.7, 0.7),
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.0, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CUBE_LARGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CubeLarge",
    spawn=sim_utils.CuboidCfg(
        size=(0.5, 0.5, 0.7),
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.05, 0.05)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CUBE_MEDIUM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CubeMedium",
    spawn=sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.7),
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CUBE_SMALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CubeSmall",
    spawn=sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.7),
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.3)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)


# ---------------------------------------------------------------------------
# Cylinder variants
# diameters: large=0.6 (r=0.3), medium=0.4 (r=0.2), small=0.2 (r=0.1)
# all 0.7 m tall
# ---------------------------------------------------------------------------

CYLINDER_XLARGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CylinderXLarge",
    spawn=sim_utils.CylinderCfg(
        radius=0.35,
        height=0.7,
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.4)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CYLINDER_LARGE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CylinderLarge",
    spawn=sim_utils.CylinderCfg(
        radius=0.25,
        height=0.7,
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.6)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CYLINDER_MEDIUM_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CylinderMedium",
    spawn=sim_utils.CylinderCfg(
        radius=0.15,
        height=0.7,
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.1, 0.8)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)

CYLINDER_SMALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/CylinderSmall",
    spawn=sim_utils.CylinderCfg(
        radius=0.05,
        height=0.7,
        rigid_props=_rigid_props(),
        mass_props=_mass(),
        collision_props=_collision(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.9)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.35)),
)


# ---------------------------------------------------------------------------
# Shape index constants — order must match assets list in ObstacleManager
# 0=cube_large, 1=cube_medium, 2=cube_small
# 3=cyl_large,  4=cyl_medium,  5=cyl_small
# ---------------------------------------------------------------------------

SHAPE_NAMES = {
    0: "cube_xlarge", 
    1: "cube_large", 
    2: "cube_medium", 
    3: "cube_small",
    4: "cylinder_xlarge",
    5: "cylinder_large", 
    6: "cylinder_medium", 
    7: "cylinder_small",
}

SHAPE_RADIUS = {
    0: 0.49,   # cube_xlarge: half-diagonal of 0.7×0.7
    1: 0.35,   # cube_large: half-diagonal of 0.5×0.5
    2: 0.21,   # cube_medium: half-diagonal of 0.3×0.3
    3: 0.07,   # cube_small: half-diagonal of 0.1×0.1
    4: 0.35,   # cylinder_xlarge: r=0.35
    5: 0.25,   # cylinder_large: r=0.25
    6: 0.15,   # cylinder_medium: r=0.15
    7: 0.1,    # cylinder_small: r=0.1
}


# ---------------------------------------------------------------------------
# Pool descriptor
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ObstacleSetCfg:

    max_obstacles: int = 8       # matches curriculum ceiling
    shapes: list[int] = dataclasses.field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7]
    )
    corridor_half_width: float = 1.5
    spawn_radius: float = 4.0
    min_obstacle_spacing: float = 0.15
    max_speed: float = 1.5