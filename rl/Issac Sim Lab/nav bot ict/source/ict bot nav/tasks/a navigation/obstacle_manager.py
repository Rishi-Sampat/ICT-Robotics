from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING
import torch
from ict_bot_nav.assets.obstacles.object_obstacle import SHAPE_RADIUS

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.assets import RigidObject
    from ict_bot_nav.assets.obstacles.object_obstacle import ObstacleSetCfg

PARK_Z      = -10.0
PARK_XY     = (1000.0, 1000.0)
NUM_SHAPES  = 8
PATH_LENGTH = 13.0
PATH_SPLIT  = 6.5


def _path_to_xy(s: float, w: float) -> tuple[float, float]:
    if s <= PATH_SPLIT:
        return -2.0 + s, 4.5 + w
    else:
        return 4.5 + w, 4.5 - (s - PATH_SPLIT)


class ObstacleManager:
    """
    Single prim per shape (8 total assets, NOT 8 slots × 8 shapes).
    Each env independently chooses which shape occupies which logical
    obstacle "slot" via self._shape_id[env, slot] = shape_id (0-7).
    _write_to_sim matches envs to the correct single prim per shape,
    exactly like the old working 6-asset version, extended to 8.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        assets: list[RigidObject],   # exactly NUM_SHAPES (8) assets — one per shape
        cfg: ObstacleSetCfg,
    ):
        assert len(assets) == NUM_SHAPES, (
            f"Expected {NUM_SHAPES} assets (one per shape), got {len(assets)}"
        )
        self.env      = env
        self._assets  = assets        # flat list of 8, index = shape_id
        self.cfg      = cfg
        self.device   = env.device
        self.num_envs = env.num_envs
        self.max_slots = cfg.max_obstacles   # logical obstacle count ceiling, e.g. 8

        n = self.num_envs
        m = self.max_slots
        self._pos      = torch.zeros(n, m, 3, device=self.device)
        self._vel      = torch.zeros(n, m, 2, device=self.device)
        self._shape_id = torch.full((n, m), -1, dtype=torch.long, device=self.device)

        self._obstacle_count = 0
        self._max_speed      = 0.0

        self._max_accel   = 1.5   # m/s² — reasonable human acceleration
        self._accel_noise = 0.5   # random perturbation to acceleration each step

    def set_curriculum_params(self, obstacle_count: int, max_speed: float) -> None:
        self._obstacle_count = max(0, min(obstacle_count, self.max_slots))
        self._max_speed      = max(0.0, max_speed)

        # Acceleration scales with speed — faster obstacles have proportionally
        # larger acceleration variance to simulate human-like movement
        # Below 0.3 m/s: near-constant (boxes, carts)
        # Above 0.5 m/s: human-like acceleration and direction changes
        if max_speed <= 0.0:
            self._accel_noise = 0.0
        elif max_speed < 0.3:
            self._accel_noise = 0.1   # small perturbation, mostly constant
        elif max_speed < 0.8:
            self._accel_noise = 0.5   # moderate — slow humans, carts
        else:
            self._accel_noise = 1.5   # full human acceleration variance

    def reset(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        park = torch.tensor([PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device)
        self._pos[env_ids]      = park
        self._shape_id[env_ids] = -1
        self._vel[env_ids]      = 0.0

        total = self._obstacle_count
        if total > 0:
            h_count = total // 2
            v_count = total - h_count

            for eid in env_ids.tolist():
                placed: list[tuple[float, float, int]] = []
                slot = 0

                # Sample unique shape_ids without replacement — one prim per shape
                shape_pool = random.sample(range(NUM_SHAPES), total)
                pool_idx   = 0

                for i in range(h_count):
                    s_min = i * (PATH_SPLIT / h_count)
                    s_max = (i + 1) * (PATH_SPLIT / h_count)
                    shape_id = shape_pool[pool_idx]
                    pool_idx += 1

                    xy = None
                    for _attempt in range(20):
                        s = random.uniform(s_min, s_max)
                        w = random.uniform(-1.0, 1.0)
                        x, y = _path_to_xy(s, w)
                        xy = self._check_placement(x, y, shape_id, placed)
                        if xy is not None:
                            break

                    if xy is not None:
                        self._pos[eid, slot, 0] = xy[0]
                        self._pos[eid, slot, 1] = xy[1]
                        self._pos[eid, slot, 2] = 0.35
                        self._vel[eid, slot, :2] = self._random_velocity()
                        self._shape_id[eid, slot] = shape_id
                        placed.append((xy[0], xy[1], shape_id))
                    slot += 1

                for i in range(v_count):
                    s_min = PATH_SPLIT + i * ((PATH_LENGTH - PATH_SPLIT) / v_count)
                    s_max = PATH_SPLIT + (i + 1) * ((PATH_LENGTH - PATH_SPLIT) / v_count)
                    shape_id = shape_pool[pool_idx]
                    pool_idx += 1

                    xy = None
                    for _attempt in range(20):
                        s = random.uniform(s_min, s_max)
                        w = random.uniform(-1.0, 1.0)
                        x, y = _path_to_xy(s, w)
                        xy = self._check_placement(x, y, shape_id, placed)
                        if xy is not None:
                            break

                    if xy is not None:
                        self._pos[eid, slot, 0] = xy[0]
                        self._pos[eid, slot, 1] = xy[1]
                        self._pos[eid, slot, 2] = 0.35
                        self._vel[eid, slot, :2] = self._random_velocity()
                        self._shape_id[eid, slot] = shape_id
                        placed.append((xy[0], xy[1], shape_id))
                    slot += 1

        self._write_to_sim(env_ids)

    def step(self) -> None:
        total = self._obstacle_count
        if total == 0:
            return

        if self._max_speed > 0.0:
            dt = self.env.physics_dt
            h_count = total // 2

            for slot in range(total):
                pos = self._pos[:, slot, :]   # [N, 3]
                vel = self._vel[:, slot, :]   # [N, 2]

                # Random acceleration perturbation — each env independently
                # This simulates direction changes and speed variation
                accel = (torch.rand_like(vel) * 2.0 - 1.0) * self._accel_noise  # [N, 2]
                vel_new = vel + accel * dt

                # Clamp to max_speed
                speed = torch.norm(vel_new, dim=-1, keepdim=True).clamp(min=1e-6)
                over_limit = speed > self._max_speed
                vel_new = torch.where(
                    over_limit,
                    vel_new / speed * self._max_speed,
                    vel_new
                )

                pos[:, :2] += vel_new * dt

                x, y = pos[:, 0], pos[:, 1]
                vx, vy = vel_new[:, 0], vel_new[:, 1]

                if slot < h_count:
                    x_lo, x_hi = -2.0, 5.5
                    y_lo, y_hi = 3.5, 5.5
                else:
                    x_lo, x_hi = 3.5, 5.5
                    y_lo, y_hi = -2.0, 5.5

                flip_x = (x < x_lo) | (x > x_hi)
                flip_y = (y < y_lo) | (y > y_hi)
                vel_new[:, 0] = torch.where(flip_x, -vx, vx)
                vel_new[:, 1] = torch.where(flip_y, -vy, vy)
                pos[:, 0] = torch.clamp(x, x_lo, x_hi)
                pos[:, 1] = torch.clamp(y, y_lo, y_hi)

                self._vel[:, slot, :] = vel_new
                self._pos[:, slot, :] = pos

        all_ids = torch.arange(self.num_envs, device=self.device)
        self._write_to_sim(all_ids)

    def _write_to_sim(self, env_ids: torch.Tensor) -> None:
        """Shape-matching write — exactly like the old working version, scaled to 8 shapes."""
        n           = len(env_ids)
        env_origins = self.env.scene.env_origins

        world_pos = self._pos[env_ids].clone()
        world_pos[:, :, 0] += env_origins[env_ids, 0].unsqueeze(1)
        world_pos[:, :, 1] += env_origins[env_ids, 1].unsqueeze(1)
        world_pos[:, :, 2] += env_origins[env_ids, 2].unsqueeze(1)

        park    = torch.tensor([PARK_XY[0], PARK_XY[1], PARK_Z], device=self.device)
        quat_id = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        for shape_id in range(NUM_SHAPES):
            asset = self._assets[shape_id]   # the ONE prim for this shape

            pos_list, vel_list = [], []
            for i in range(n):
                eid       = env_ids[i]
                shape_row = self._shape_id[eid]                       # [max_slots]
                matches   = (shape_row == shape_id).nonzero(as_tuple=False)

                if len(matches) > 0:
                    slot = int(matches[0, 0])   # first slot using this shape, this env
                    pos_list.append(world_pos[i, slot])
                    vel_list.append(self._vel[eid, slot])
                else:
                    pos_list.append(park)
                    vel_list.append(torch.zeros(2, device=self.device))

            pos_t = torch.stack(pos_list)
            vel_t = torch.stack(vel_list)
            pose  = torch.cat([pos_t, quat_id.unsqueeze(0).expand(n, -1)], dim=-1)
            twist = torch.zeros(n, 6, device=self.device)
            twist[:, :2] = vel_t

            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(twist, env_ids=env_ids)

    def _random_velocity(self) -> torch.Tensor:
        if self._max_speed == 0.0:
            return torch.zeros(2, device=self.device)
        speed = random.uniform(0.1, self._max_speed)
        angle = random.uniform(0.0, 2.0 * math.pi)
        return torch.tensor(
            [speed * math.cos(angle), speed * math.sin(angle)], device=self.device
        )

    def _check_placement(self, x, y, shape_id, placed):
        own_r = SHAPE_RADIUS[shape_id]
        for px, py, pid in placed:
            if math.hypot(x - px, y - py) < own_r + SHAPE_RADIUS[pid] + self.cfg.min_obstacle_spacing:
                return None
        return (x, y)