import argparse
import os
import pickle

import genesis as gs
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor

from src.go2.go2_sb3_env import Go2GymEnv
from src.configs import get_cfgs
from src.configs.seed import set_global_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-sb3-sac")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    set_global_seed()

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()[:4]

    with open(os.path.join(log_dir, "cfgs_sb3.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], f)

    train_env = Go2GymEnv(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device="cuda:0" if args.device == "cuda" else "cpu",
        show_viewer=False,
    )

    eval_env = Go2GymEnv(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        device="cuda:0" if args.device == "cuda" else "cpu",
        show_viewer=False,
    )
    eval_env = Monitor(eval_env)

    model = SAC(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(log_dir, "tb"),
        device=args.device,
        learning_rate=3e-4,
        batch_size=256,
        train_freq=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        use_sde=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=log_dir,
        name_prefix="sac_go2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
        verbose=1,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval_logs"),
        eval_freq=10_000,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    model.save(os.path.join(log_dir, "sac_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
