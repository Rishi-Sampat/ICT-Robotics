from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ---------------------------------------------------------------------------
# Schedule definition
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CurriculumLevel:
    obstacle_count: int
    max_speed: float
    promote_threshold: float = 0.70
    demote_threshold: float  = 0.40

OBSTACLE_SCHEDULE: list[CurriculumLevel] = [
    CurriculumLevel(obstacle_count=0, max_speed=0.0, promote_threshold=0.60),
    CurriculumLevel(obstacle_count=1, max_speed=0.0),
    CurriculumLevel(obstacle_count=2, max_speed=0.1),
    CurriculumLevel(obstacle_count=3, max_speed=0.3),
    CurriculumLevel(obstacle_count=4, max_speed=0.5),
    CurriculumLevel(obstacle_count=5, max_speed=0.8),
    CurriculumLevel(obstacle_count=6, max_speed=1.0),
    CurriculumLevel(obstacle_count=7, max_speed=1.3),
    CurriculumLevel(obstacle_count=8, max_speed=1.5),
]


# ---------------------------------------------------------------------------
# Curriculum configuration dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ObstacleCurriculumCfg:
    """Configuration for the obstacle curriculum term."""

    eval_window: int                    = 1000  
    consecutive_windows_to_promote: int = 2
    success_key: str                    = "goal_reached"
    play_level: int | None              = None
    transition_cooldown: int            = 5000   # episodes


# ---------------------------------------------------------------------------
# Term function
# ---------------------------------------------------------------------------

def obstacle_curriculum_term(
    env: "ManagerBasedRLEnv",
    env_ids,
    cfg: ObstacleCurriculumCfg,
) -> dict:
    """Evaluate success rate and handle clean level transitions."""

    if not hasattr(env, "_obs_curr_level"):
        env._obs_curr_level       = 0
        env._obs_curr_consecutive = 0
        env._obs_curr_successes   = []
        env._obs_curr_cooldown    = 0   

    if cfg.play_level is not None:
        if env._obs_curr_level != cfg.play_level:
            env._obs_curr_level = cfg.play_level
            _apply_curriculum(env, cfg.play_level)
        return {"obstacle_level": env._obs_curr_level}

    import torch

    ep_extras = env.extras.get("episode", {})
    successes  = ep_extras.get(cfg.success_key, None)
    n_new = len(env_ids)

    if successes is not None:
        if isinstance(successes, torch.Tensor):
            if successes.ndim == 0:
                env._obs_curr_successes.extend([successes.item()] * n_new)
            elif successes.shape[0] == env.num_envs:
                env._obs_curr_successes.extend(successes[env_ids].tolist())
            else:
                env._obs_curr_successes.extend(successes.tolist())
        else:
            env._obs_curr_successes.extend([float(successes)] * n_new)

    env._obs_curr_cooldown = max(0, env._obs_curr_cooldown - n_new)

    if len(env._obs_curr_successes) < cfg.eval_window:
        return {"obstacle_level": env._obs_curr_level}

    window       = env._obs_curr_successes[-cfg.eval_window:]
    success_rate = sum(window) / len(window)
    level        = OBSTACLE_SCHEDULE[env._obs_curr_level]

    if env._obs_curr_cooldown == 0:
        if success_rate >= level.promote_threshold:
            env._obs_curr_consecutive += 1
            if env._obs_curr_consecutive >= cfg.consecutive_windows_to_promote:
                if env._obs_curr_level < len(OBSTACLE_SCHEDULE) - 1:
                    env._obs_curr_level      += 1
                    env._obs_curr_consecutive = 0
                    env._obs_curr_cooldown    = cfg.transition_cooldown
                    _apply_curriculum(env, env._obs_curr_level)
        elif success_rate < level.demote_threshold:
            if env._obs_curr_level > 0:
                env._obs_curr_level      -= 1
                env._obs_curr_consecutive = 0
                env._obs_curr_cooldown    = cfg.transition_cooldown
                _apply_curriculum(env, env._obs_curr_level)
        else:
            env._obs_curr_consecutive = 0

    env._obs_curr_successes = env._obs_curr_successes[-(cfg.eval_window * 4):]

    return {
        "obstacle_level":        env._obs_curr_level,
        "obstacle_success_rate": success_rate,
        "obstacle_total":        OBSTACLE_SCHEDULE[env._obs_curr_level].obstacle_count,
        "obstacle_max_speed":    OBSTACLE_SCHEDULE[env._obs_curr_level].max_speed,
        "obstacle_cooldown":     env._obs_curr_cooldown,
    }


def _apply_curriculum(env, level: int) -> None:
    schedule = OBSTACLE_SCHEDULE[level]
    env.obstacle_manager.set_curriculum_params(
        obstacle_count=schedule.obstacle_count,
        max_speed=schedule.max_speed,
    )