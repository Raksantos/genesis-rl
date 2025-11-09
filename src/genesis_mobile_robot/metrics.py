import numpy as np


class EpisodeMetrics:
    def __init__(self, compute_clearance=False, obstacles=None):
        self.compute_clearance = compute_clearance
        self.obstacles = obstacles or []
        self.reset()

    def reset(self):
        self.success = 0
        self.collisions = 0
        self.time = 0.0
        self.path_length = 0.0
        self.min_clearance = float("inf")
        self.prev_pos = None
        self.prev_action = None
        self.smoothness = 0.0
        self.steps = 0

    def update(self, info: dict):
        pos = info.get("pos")
        dt = info.get("dt", 0.0)
        collided = info.get("collided", False)
        action = info.get("action")

        self.time += dt
        self.steps += 1

        if collided:
            self.collisions += 1

        if pos is not None:
            pos = np.array(pos, dtype=float)
            if self.prev_pos is not None:
                self.path_length += float(np.linalg.norm(pos - self.prev_pos))
            self.prev_pos = pos

            if self.compute_clearance and self.obstacles:
                d = self._compute_min_clearance(pos)
                self.min_clearance = min(self.min_clearance, d)

        if action is not None:
            action = np.array(action, dtype=float)
            if self.prev_action is not None:
                delta = action - self.prev_action
                self.smoothness += float(np.dot(delta, delta))
            self.prev_action = action

    def _compute_min_clearance(self, pos2d: np.ndarray) -> float:
        dists = []
        for ob in self.obstacles:
            ob_pos = np.array(ob[:2], dtype=float)
            dists.append(np.linalg.norm(pos2d - ob_pos))
        return min(dists) if dists else float("inf")

    def finalize(self, info: dict):
        if info.get("reached_goal", False):
            self.success = 1

        if self.min_clearance == float("inf"):
            self.min_clearance = None

        return {
            "success_rate": self.success,
            "number_of_collisions": self.collisions,
            "time_to_goal": self.time,
            "path_length": self.path_length,
            "minimum_clearance": self.min_clearance,
            "trajectory_smoothness": self.smoothness,
            "steps": self.steps,
        }
