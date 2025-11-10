from __future__ import annotations

import math

import genesis as gs
import gymnasium as gym
import numpy as np


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class GenesisDiffDriveGoalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        xml_path: str,
        arena_half: float = 5.0,
        max_steps: int = 300,
        frame_skip: int = 1,
        wheel_separation: float = 0.36,
        max_wheel_vel: float = 6.0,
        show_viewer: bool = False,
    ):
        super().__init__()

        self.arena_half = arena_half
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.wheel_separation = wheel_separation
        self.max_wheel_vel = max_wheel_vel
        self.dt = 0.01

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        high = np.array(
            [
                arena_half,
                arena_half,
                1.0,
                1.0,
                arena_half,
                arena_half,
                2 * arena_half,
                2 * arena_half,
                2 * math.sqrt(2) * arena_half,
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.scene = gs.Scene(show_viewer=show_viewer)
        self.scene.add_entity(gs.morphs.Plane())

        h = arena_half
        t = 0.15
        wh = 1.5

        def add_wall(lower, upper):
            self.scene.add_entity(gs.morphs.Box(lower=lower, upper=upper, fixed=True))

        add_wall(lower=(-h, h - t, 0.0), upper=(h, h, wh))
        add_wall(lower=(-h, -h, 0.0), upper=(h, -h + t, wh))
        add_wall(lower=(h - t, -h, 0.0), upper=(h, h, wh))
        add_wall(lower=(-h, -h, 0.0), upper=(-h + t, h, wh))

        self.robot = self.scene.add_entity(gs.morphs.MJCF(file=xml_path, pos=(-4.0, -4.0, 0.0)))

        self.goal_pos = np.array([4.0, 4.0, 0.05], dtype=np.float32)
        self.scene.add_entity(
            gs.morphs.Box(pos=tuple(self.goal_pos), size=(0.15, 0.15, 0.08), fixed=True)
        )

        self.scene.build()

        self.x = -4.0
        self.y = -4.0
        self.yaw = 0.0

        self.step_count = 0
        self.prev_dist = None

    def _sync_visual_robot(self):
        quat = yaw_to_quat(self.yaw)
        self.robot.set_pos(np.array([self.x, self.y, 0.0], dtype=np.float32))
        self.robot.set_quat(quat)

    def _get_obs(self):
        gx, gy, _ = self.goal_pos
        dx = gx - self.x
        dy = gy - self.y
        dist = math.hypot(dx, dy)
        obs = np.array(
            [
                self.x,
                self.y,
                math.cos(self.yaw),
                math.sin(self.yaw),
                gx,
                gy,
                dx,
                dy,
                dist,
            ],
            dtype=np.float32,
        )
        return obs, dist

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.x = -4.0
        self.y = -4.0
        self.yaw = 0.0
        self._sync_visual_robot()

        self.scene.step()

        self.step_count = 0
        obs, dist = self._get_obs()
        self.prev_dist = dist
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        wl = float(action[0]) * self.max_wheel_vel
        wr = float(action[1]) * self.max_wheel_vel

        v = (wr + wl) * 0.5 * 0.1
        omega = (wr - wl) / self.wheel_separation

        for _ in range(self.frame_skip):
            self.x += v * math.cos(self.yaw) * self.dt
            self.y += v * math.sin(self.yaw) * self.dt
            self.yaw += omega * self.dt

        self._sync_visual_robot()

        self.scene.step()

        obs, dist = self._get_obs()

        reward = 0.0
        if self.prev_dist is not None:
            reward += (self.prev_dist - dist) * 5.0
        reward -= 0.001 * (abs(wl) + abs(wr))
        self.prev_dist = dist

        self.step_count += 1

        reached_goal = dist < 0.3
        terminated = reached_goal
        truncated = self.step_count >= self.max_steps

        info = {
            "dist": dist,
            "reached_goal": reached_goal,
            "collided": False,
            "pos": np.array([self.x, self.y], dtype=np.float32),
            "dt": self.dt * self.frame_skip,
            "action": np.array([wl, wr], dtype=np.float32),
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
