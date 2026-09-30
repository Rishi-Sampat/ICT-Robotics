# ict_bot/assets/markers.py
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

TARGET_CONE_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Target_cone",
    markers={
        "cone": sim_utils.ConeCfg(
            radius=0.1,
            height=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)