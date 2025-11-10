from dataclasses import dataclass
import math
import numpy as np


@dataclass
class DWAConfig:
    max_speed: float = 1.5
    min_speed: float = 0.0
    max_yaw_rate: float = math.radians(120.0)
    max_accel: float = 1.0
    max_delta_yaw_rate: float = math.radians(180.0)
    v_reso: float = 0.01
    yaw_rate_reso: float = math.radians(2.0)
    dt: float = 0.1
    predict_time: float = 2.5

    to_goal_cost_gain: float = 2.0
    speed_cost_gain: float = 0.5
    obstacle_cost_gain: float = 0.5

    robot_radius: float = 0.25
