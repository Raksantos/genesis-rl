from pathlib import Path
from src.genesis_mobile_robot.env_diff_drive import GenesisDiffDriveGoalEnv
from src.algorithms.dwa import DWAConfig, DWAPlanner
import numpy as np
import math

def run_dwa():
    root = Path(__file__).resolve().parents[2]
    xml_path = root / "genesis-rl" / "xml" / "mobile_base" / "diff_drive.xml"
    env = GenesisDiffDriveGoalEnv(str(xml_path), show_viewer=True)

    cfg = DWAConfig()
    planner = DWAPlanner(cfg)

    obs, _ = env.reset()

    obstacles = [
        (-5.0, y) for y in np.linspace(-5.0, 5.0, 10)
    ] + [
        (5.0, y) for y in np.linspace(-5.0, 5.0, 10)
    ] + [
        (x, -5.0) for x in np.linspace(-5.0, 5.0, 10)
    ] + [
        (x, 5.0) for x in np.linspace(-5.0, 5.0, 10)
    ]

    done = False
    last_info = {}

    x = float(obs[0])
    y = float(obs[1])
    yaw = math.atan2(float(obs[3]), float(obs[2]))
    x_dwa = np.array([x, y, yaw, 0.0, 0.0], dtype=float)

    while not done:
        gx = env.goal_pos[0]
        gy = env.goal_pos[1]

        (v, w), _ = planner.plan(x_dwa, (gx, gy), obstacles)

        wheel_sep = 0.36
        wl = v - (w * wheel_sep / 2.0)
        wr = v + (w * wheel_sep / 2.0)

        action = np.array([wl, wr], dtype=np.float32)
        obs, _, terminated, truncated, info = env.step(action)

        last_info = info

        x = float(obs[0])
        y = float(obs[1])
        yaw = math.atan2(float(obs[3]), float(obs[2]))

        x_dwa = np.array([x, y, yaw, v, w], dtype=float)

        done = terminated or truncated

    print("episódio terminou; dist final:", last_info.get("dist"))

if __name__ == "__main__":
    run_dwa()