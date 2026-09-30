from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor


def time_out(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Episode terminates if max episode length is reached."""
    return env.episode_length_buf >= env.max_episode_length