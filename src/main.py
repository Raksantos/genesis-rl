# src/genesis_mobile_robot/run_episode_curriculum.py

import math
import numpy as np
import logging
from datetime import datetime
from pathlib import Path

from src.genesis_mobile_robot.env_diff_drive import GenesisDiffDriveGoalEnv
from src.algorithms.a_star import astar, world_to_grid, grid_to_world
from src.genesis_mobile_robot.map_generator import build_empty_grid, add_random_blocks
from src.algorithms.dwa import DWAPlanner, DWAConfig
from src.helpers.metrics import EpisodeMetrics
import genesis as gs


# ================================
# LOGGER ENXUTO (só arquivo)
# ================================
LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(file_handler)


def run_episode(env, show_viewer=False, dwa_params=None, seed=None):
    metrics = EpisodeMetrics()

    obs, _ = env.reset()

    x_min, x_max = -5.0, 5.0
    y_min, y_max = -5.0, 5.0
    res = 0.25

    grid = build_empty_grid(x_min, x_max, y_min, y_max, res)
    grid = add_random_blocks(grid, x_min, y_min, res, num_blocks=4)

    start_x = float(obs[0])
    start_y = float(obs[1])
    goal_x = 4.0
    goal_y = 4.0

    start_cell = world_to_grid(start_x, start_y, x_min, y_min, res)
    goal_cell = world_to_grid(goal_x, goal_y, x_min, y_min, res)

    path_cells = astar(grid, start_cell, goal_cell)
    if not path_cells:
        result = {
            "success_rate": 0,
            "number_of_collisions": 0,
            "time_to_goal": env.max_steps * getattr(env, "dt", 0.01),
            "path_length": 0.0,
            "minimum_clearance": None,
            "trajectory_smoothness": 0.0,
            "steps": 0,
        }
        logger.info(f"[episodio] {result}")
        return result

    waypoints = [grid_to_world(i, j, x_min, y_min, res) for (i, j) in path_cells]

    cfg = DWAConfig()
    if dwa_params:
        for k, v in dwa_params.items():
            setattr(cfg, k, v)
    planner = DWAPlanner(cfg)

    x = start_x
    y = start_y
    yaw = math.atan2(float(obs[3]), float(obs[2]))
    x_dwa = np.array([x, y, yaw, 0.0, 0.0], dtype=float)

    wp_idx = 0
    done = False
    last_info = {}

    while not done and wp_idx < len(waypoints):
        gx, gy = waypoints[wp_idx]
        if math.hypot(gx - x_dwa[0], gy - x_dwa[1]) < 0.3:
            wp_idx += 1
            continue

        obstacles = []
        (v, w), _ = planner.plan(x_dwa, (gx, gy), obstacles)

        wheel_sep = 0.36
        wl = v - (w * wheel_sep / 2.0)
        wr = v + (w * wheel_sep / 2.0)
        action = np.array([wl, wr], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        metrics.update(info)
        last_info = info

        x = float(obs[0])
        y = float(obs[1])
        yaw = math.atan2(float(obs[3]), float(obs[2]))
        x_dwa = np.array([x, y, yaw, v, w], dtype=float)

        done = terminated or truncated

    result = metrics.finalize(last_info)
    logger.info(f"[episodio] {result}")
    return result


# def main():
#     N = 20
#     results = []

#     gs.init(backend=gs.gpu)

#     xml_path = (
#         Path(__file__).resolve().parents[2]
#         / "genesis-rl"
#         / "xml"
#         / "mobile_base"
#         / "diff_drive.xml"
#     )
#     env = GenesisDiffDriveGoalEnv(str(xml_path), show_viewer=False)

#     for i in range(N):
#         res = run_episode(env, show_viewer=False)
#         results.append(res)

#     success = sum(r["success_rate"] for r in results) / N
#     avg_time = sum(r["time_to_goal"] for r in results) / N
#     avg_coll = sum(r["number_of_collisions"] for r in results) / N

#     summary = {
#         "success_medio": success,
#         "tempo_medio": avg_time,
#         "colisoes_medias": avg_coll,
#     }
#     logger.info(f"[resumo] {summary}")

#     print("==== resumo currículo ====")
#     print(summary)

def main():
    gs.init(backend=gs.gpu)  # ou gs.cpu

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(5.0, 0.0, 3.0),
            camera_lookat=(0.0, 0.0, 0.5),
        ),
    )

    scene.add_entity(gs.morphs.Plane())

    half = 5.0
    t = 0.15
    h = 1.5
    scene.add_entity(gs.morphs.Box(lower=(-half, half - t, 0), upper=(half, half, h), fixed=True))
    scene.add_entity(gs.morphs.Box(lower=(-half, -half, 0), upper=(half, -half + t, h), fixed=True))
    scene.add_entity(gs.morphs.Box(lower=(half - t, -half, 0), upper=(half, half, h), fixed=True))
    scene.add_entity(gs.morphs.Box(lower=(-half, -half, 0), upper=(-half + t, half, h), fixed=True))

    project_root = Path(__file__).resolve().parents[2]
    go2_xml = project_root / "genesis-rl" / "xml" / "go2" / "go2.xml"

    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(go2_xml),
            pos=(0.0, 0.0, 0.0),
        )
    )

    scene.add_entity(
        gs.morphs.Box(
            pos=(3.0, 3.0, 0.05),
            size=(0.15, 0.15, 0.08),
            fixed=True,
        )
    )

    scene.build()

    for _ in range(2000):
        scene.step()


if __name__ == "__main__":
    main()
