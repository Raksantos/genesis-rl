import argparse
import os
import pickle

import genesis as gs
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

from src.go2.go2_sb3_env import Go2GymEnv
from src.configs import get_cfgs
from src.helpers import EarlyStopCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-sb3-td3")

    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    backend = (
        gs.constants.backend.gpu if args.device == "cuda" else gs.constants.backend.cpu
    )
    gs.init(backend=backend, logging_level="Warning")

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()[:4]

    with open(os.path.join(log_dir, "cfgs_sb3.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], f)

    env = Go2GymEnv(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device="cuda:0" if args.device == "cuda" else "cpu",
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=0.0,
        sigma=0.1 * np.ones(n_actions),
    )

    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb"),
        device=args.device,
        learning_rate=3e-4,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        buffer_size=1_000_000,
        action_noise=action_noise,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=log_dir,
        name_prefix="td3_go2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    early_stop_callback = EarlyStopCallback(patience=20, min_delta=1.0, verbose=1)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, early_stop_callback],
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "td3_final"))

    env.close()


if __name__ == "__main__":
    main()
