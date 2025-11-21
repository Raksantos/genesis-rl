import argparse
import os
import pickle

import genesis as gs
import numpy as np
import torch
from stable_baselines3 import SAC

from src.go2.go2_sb3_env import Go2GymEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-sb3-sac")
    parser.add_argument("--model_path", type=str, default="sac_final.zip")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    backend = gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    gs.init(backend=backend)

    log_dir = os.path.join("logs", args.exp_name)

    env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(
        open(os.path.join(log_dir, "cfgs_sb3.pkl"), "rb")
    )

    reward_cfg["reward_scales"] = {}
    env_cfg["termination_if_roll_greater_than"] = 50
    env_cfg["termination_if_pitch_greater_than"] = 50

    env = Go2GymEnv(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device="cuda:0" if args.device == "cuda" else "cpu",
        show_viewer=True,
    )

    model = SAC.load(os.path.join(log_dir, args.model_path), device=args.device)

    obs, _ = env.reset()
    t = 0
    lin_x_range = [0.5, 4.0]

    while True:
        action, _ = model.predict(obs, deterministic=True)

        lin_x = (
            lin_x_range[0]
            + (lin_x_range[1] - lin_x_range[0])
            * (np.sin(2 * np.pi * t / 600) + 1)
            / 2
        )
        env.go2_env.commands = torch.tensor([[lin_x, 0.0, 0.0]], device=env.device)

        obs, _, terminated, truncated, info = env.step(action)
        t += 1
        if terminated or truncated:
            t = 0
            obs, _ = env.reset()


if __name__ == "__main__":
    main()