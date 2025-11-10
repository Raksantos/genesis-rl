import numpy as np
from .dwa_config import DWAConfig
import math


class DWAPlanner:
    def __init__(self, config: DWAConfig):
        self.cfg = config

    def motion(self, x, u, dt):
        """Modelo uniciclo.
        x = [x, y, yaw, v, w]
        u = [v, w]
        """
        x_new = x.copy()
        v, w = u
        x_new[2] += w * dt
        x_new[0] += v * math.cos(x_new[2]) * dt
        x_new[1] += v * math.sin(x_new[2]) * dt
        x_new[3] = v
        x_new[4] = w
        return x_new

    def calc_dynamic_window(self, x):
        cfg = self.cfg

        Vs = [cfg.min_speed, cfg.max_speed, -cfg.max_yaw_rate, cfg.max_yaw_rate]

        Vd = [
            x[3] - cfg.max_accel * cfg.dt,
            x[3] + cfg.max_accel * cfg.dt,
            x[4] - cfg.max_delta_yaw_rate * cfg.dt,
            x[4] + cfg.max_delta_yaw_rate * cfg.dt,
        ]

        dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]
        return dw

    def predict_trajectory(self, x_init, v, w):
        cfg = self.cfg
        x = x_init.copy()
        traj = [x.copy()]
        time = 0.0
        while time <= cfg.predict_time:
            x = self.motion(x, [v, w], cfg.dt)
            traj.append(x.copy())
            time += cfg.dt
        return np.array(traj)

    def calc_obstacle_cost(self, traj, obstacles):
        cfg = self.cfg
        if obstacles is None or len(obstacles) == 0:
            return 0.0

        min_r = float("inf")
        for ob in obstacles:
            dx = traj[:, 0] - ob[0]
            dy = traj[:, 1] - ob[1]
            dist = np.hypot(dx, dy)
            min_r = min(min_r, dist.min())

        if min_r <= cfg.robot_radius:
            return float("inf")
        return 1.0 / min_r

    def calc_to_goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        return math.hypot(dx, dy)

    def plan(self, x, goal, obstacles):
        """
        x: estado atual [x, y, yaw, v, w]
        goal: (gx, gy)
        obstacles: lista [(ox, oy), ...]
        """
        cfg = self.cfg
        dw = self.calc_dynamic_window(x)

        best_u = [0.0, 0.0]
        best_traj = None
        min_cost = float("inf")

        for v in np.arange(dw[0], dw[1] + cfg.v_reso, cfg.v_reso):
            for w in np.arange(dw[2], dw[3] + cfg.yaw_rate_reso, cfg.yaw_rate_reso):
                traj = self.predict_trajectory(x, v, w)

                to_goal_cost = cfg.to_goal_cost_gain * self.calc_to_goal_cost(traj, goal)
                speed_cost = cfg.speed_cost_gain * (cfg.max_speed - traj[-1, 3])
                ob_cost = cfg.obstacle_cost_gain * self.calc_obstacle_cost(traj, obstacles)

                final_cost = to_goal_cost + speed_cost + ob_cost

                if final_cost < min_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_traj = traj

        return best_u, best_traj
