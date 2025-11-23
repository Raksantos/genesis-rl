import argparse
import os
import pickle

import numpy as np
import torch
import genesis as gs

from src.go2 import Go2Env
from rsl_rl.runners import OnPolicyRunner


def build_env(env_cfg, obs_cfg, reward_cfg, command_cfg, device, show_viewer=True):
    reward_cfg["reward_scales"] = {}

    env_cfg["termination_if_roll_greater_than"] = 50
    env_cfg["termination_if_pitch_greater_than"] = 50

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=show_viewer,
    )
    return env


def load_ppo_policy(env, train_cfg, log_dir, ckpt_iter: int, device: str):
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=device)
    resume_path = os.path.join(log_dir, f"model_{ckpt_iter}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=device)

    @torch.no_grad()
    def act_fn(obs: torch.Tensor) -> torch.Tensor:
        return policy(obs.to(device))

    return act_fn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-ppo")
    parser.add_argument("-a", "--algo", type=str, choices=["ppo", "sac"], default="ppo")
    parser.add_argument("--ckpt", type=int, default=200)
    parser.add_argument("--sac_step", type=int, default=100000)
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"]
    )
    args = parser.parse_args()

    gs.init(backend=gs.constants.backend.gpu)

    log_dir = f"logs/{args.exp_name}"

    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
        open(os.path.join(log_dir, "cfgs.pkl"), "rb")
    )

    env = build_env(
        env_cfg, obs_cfg, reward_cfg, command_cfg, args.device, show_viewer=True
    )

    act_fn = load_ppo_policy(env, train_cfg, log_dir, args.ckpt, args.device)

    obs, _ = env.reset()

    env.commands = torch.tensor([[0.5, 0.0, 0.0]], device=args.device)
    t = 0
    lin_x_range = [0.5, 4.0]

    with torch.no_grad():
        while True:
            actions = act_fn(obs)
            lin_x = (
                lin_x_range[0]
                + (lin_x_range[1] - lin_x_range[0])
                * (np.sin(2 * np.pi * t / 600) + 1)
                / 2
            )
            lin_x = float(lin_x)
            print(lin_x)

            env.commands = torch.tensor([[lin_x, 0.0, 0.0]], device=args.device)
            obs, _, dones, _ = env.step(actions)
            t += 1
            if dones.any():
                t = 0
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
