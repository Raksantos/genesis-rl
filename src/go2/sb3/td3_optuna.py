import argparse
import os
import pickle
import glob

import optuna
import numpy as np
import genesis as gs
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from src.go2.go2_sb3_env import Go2GymEnv
from src.configs import get_cfgs
from src.configs.seed import set_global_seed


def evaluate_td3_policy(env, model, n_episodes: int = 10) -> float:
    """
    Evaluate a TD3 policy by running n_episodes and returning the mean return.
    """
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated

        returns.append(episode_return)

    return float(np.mean(returns)) if len(returns) > 0 else 0.0


def find_latest_checkpoint(trial_log_dir: str) -> tuple[str | None, int]:
    """
    Find the latest checkpoint in the trial directory.
    Returns (checkpoint_path, timestep) or (None, 0) if no checkpoint found.
    SB3 saves checkpoints as td3_go2_<timesteps>_steps.zip
    """
    if not os.path.exists(trial_log_dir):
        return None, 0

    checkpoint_files = glob.glob(os.path.join(trial_log_dir, "td3_go2_*_steps.zip"))

    if not checkpoint_files:
        final_model = os.path.join(trial_log_dir, "td3_final.zip")
        if os.path.exists(final_model):
            return final_model, -1
        return None, 0

    timesteps = []
    for f in checkpoint_files:
        try:
            basename = os.path.basename(f)
            timestep_str = basename.replace("td3_go2_", "").replace("_steps.zip", "")
            timestep = int(timestep_str)
            timesteps.append((timestep, f))
        except ValueError:
            continue

    if not timesteps:
        return None, 0

    timesteps.sort(key=lambda x: x[0], reverse=True)
    latest_timestep, latest_path = timesteps[0]
    return latest_path, latest_timestep


def objective(trial, args):
    """
    Optuna objective function that suggests hyperparameters, trains TD3, and returns evaluation metric.
    Supports resuming from checkpoints if a trial was interrupted.
    Note: set_global_seed() should be called once in main(), not here, to avoid reinitializing Genesis.
    """

    trial_log_dir = os.path.join("logs", args.exp_name, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    checkpoint_path, checkpoint_timestep = find_latest_checkpoint(trial_log_dir)
    resume_from_checkpoint = checkpoint_path is not None and args.resume_trials

    if resume_from_checkpoint:
        print(
            f"Trial {trial.number}: Resuming from checkpoint at timestep {checkpoint_timestep}"
        )

        try:
            with open(os.path.join(trial_log_dir, "cfgs_sb3.pkl"), "rb") as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)
        except FileNotFoundError:
            print(f"Trial {trial.number}: Config file not found, starting from scratch")
            resume_from_checkpoint = False
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()[:4]

    if not resume_from_checkpoint:
        with open(os.path.join(trial_log_dir, "cfgs_sb3.pkl"), "wb") as f:
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

    if resume_from_checkpoint and checkpoint_timestep == -1:
        print(f"Trial {trial.number}: Final model exists, evaluating...")
        try:
            model = TD3.load(checkpoint_path, device=args.device)
            mean_return = evaluate_td3_policy(eval_env, model, args.n_eval_episodes)
            trial.set_user_attr("mean_return", mean_return)
            train_env.close()
            eval_env.close()
            return mean_return
        except Exception as e:
            print(f"Trial {trial.number}: Error evaluating final model: {e}")
            train_env.close()
            eval_env.close()
            return float("-inf")

    if not resume_from_checkpoint:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16])
        gradient_steps = trial.suggest_int("gradient_steps", 1, 4)
        gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=False)
        tau = trial.suggest_float("tau", 0.001, 0.01, log=False)

        policy_delay = trial.suggest_int("policy_delay", 1, 4)
        target_policy_noise = trial.suggest_float(
            "target_policy_noise", 0.1, 0.5, log=False
        )
        target_noise_clip = trial.suggest_float(
            "target_noise_clip", 0.3, 0.7, log=False
        )

        action_noise_sigma = trial.suggest_float(
            "action_noise_sigma", 0.05, 0.3, log=False
        )

        net_arch_type = trial.suggest_categorical(
            "net_arch_type", ["small", "medium", "large"]
        )
        if net_arch_type == "small":
            net_arch = [64, 64]
        elif net_arch_type == "medium":
            net_arch = [256, 256]
        else:
            net_arch = [512, 512]

        buffer_size = trial.suggest_categorical(
            "buffer_size", [100_000, 300_000, 1_000_000]
        )

        learning_starts = trial.suggest_int("learning_starts", 100, 10_000, step=100)

        hyperparams = {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "train_freq": train_freq,
            "gradient_steps": gradient_steps,
            "gamma": gamma,
            "tau": tau,
            "policy_delay": policy_delay,
            "target_policy_noise": target_policy_noise,
            "target_noise_clip": target_noise_clip,
            "action_noise_sigma": action_noise_sigma,
            "net_arch": net_arch,
            "buffer_size": buffer_size,
            "learning_starts": learning_starts,
        }

        with open(os.path.join(trial_log_dir, "hyperparams.pkl"), "wb") as f:
            pickle.dump(hyperparams, f)
    else:
        try:
            with open(os.path.join(trial_log_dir, "hyperparams.pkl"), "rb") as f:
                hyperparams = pickle.load(f)
        except FileNotFoundError:
            print(f"Trial {trial.number}: Hyperparams file not found, using defaults")
            hyperparams = {
                "learning_rate": 3e-4,
                "batch_size": 256,
                "train_freq": 1,
                "gradient_steps": 1,
                "gamma": 0.99,
                "tau": 0.005,
                "policy_delay": 2,
                "target_policy_noise": 0.2,
                "target_noise_clip": 0.5,
                "action_noise_sigma": 0.1,
                "net_arch": [256, 256],
                "buffer_size": 1_000_000,
                "learning_starts": 100,
            }

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=0.0,
        sigma=hyperparams["action_noise_sigma"] * np.ones(n_actions),
    )

    if resume_from_checkpoint and checkpoint_timestep > 0:
        model = TD3.load(checkpoint_path, env=train_env, device=args.device)
        remaining_timesteps = max(0, args.total_timesteps - checkpoint_timestep)
        print(
            f"Trial {trial.number}: Loaded checkpoint, {remaining_timesteps} timesteps remaining"
        )
    else:
        model = TD3(
            "MlpPolicy",
            train_env,
            verbose=0,
            tensorboard_log=os.path.join(trial_log_dir, "tb"),
            device=args.device,
            learning_rate=hyperparams["learning_rate"],
            batch_size=hyperparams["batch_size"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams["gradient_steps"],
            gamma=hyperparams["gamma"],
            tau=hyperparams["tau"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["learning_starts"],
            action_noise=action_noise,
            policy_delay=hyperparams["policy_delay"],
            target_policy_noise=hyperparams["target_policy_noise"],
            target_noise_clip=hyperparams["target_noise_clip"],
            policy_kwargs=dict(net_arch=hyperparams["net_arch"]),
        )
        remaining_timesteps = args.total_timesteps

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=trial_log_dir,
        name_prefix="td3_go2",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=args.max_no_improvement_evals,
        min_evals=args.min_evals,
        verbose=0,
    )

    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        best_model_save_path=os.path.join(trial_log_dir, "best_model"),
        log_path=os.path.join(trial_log_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=0,
    )

    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=False,
                reset_num_timesteps=False if resume_from_checkpoint else True,
            )

        model.save(os.path.join(trial_log_dir, "td3_final"))

        mean_return = evaluate_td3_policy(
            eval_env, model, n_episodes=args.n_eval_episodes
        )

        trial.set_user_attr("mean_return", mean_return)

        train_env.close()
        eval_env.close()

        return mean_return

    except KeyboardInterrupt:
        print(f"\nTrial {trial.number}: Interrupted by user. Checkpoint saved.")

        model.save(os.path.join(trial_log_dir, "td3_interrupted"))
        train_env.close()
        eval_env.close()
        raise

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback

        traceback.print_exc()
        train_env.close()
        eval_env.close()

        return float("-inf")


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for TD3 with Stable-Baselines3"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-sb3-td3-optuna")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=200_000,
        help="Total timesteps per trial",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=50_000,
        help="Frequency of checkpoint saving",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=10_000,
        help="Frequency of evaluation",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=5,
        help="Number of episodes for evaluation",
    )
    parser.add_argument(
        "--max_no_improvement_evals",
        type=int,
        default=10,
        help="Max evaluations without improvement before early stopping",
    )
    parser.add_argument(
        "--min_evals",
        type=int,
        default=10,
        help="Minimum evaluations before early stopping can trigger",
    )
    parser.add_argument(
        "--n_trials", type=int, default=50, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=None,
        help="Name for Optuna study (for resuming)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Storage URL for Optuna study (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--prune_trials",
        action="store_true",
        help="Enable pruning of unpromising trials",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="maximize",
        choices=["maximize", "minimize"],
        help="Direction of optimization",
    )
    parser.add_argument(
        "--resume_trials",
        action="store_true",
        help="Resume incomplete trials from checkpoints if they exist",
    )
    args = parser.parse_args()

    set_global_seed()

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    if args.study_name is None:
        args.study_name = args.exp_name

    storage_url = args.storage or f"sqlite:///{os.path.join(log_dir, 'optuna.db')}"

    pruner = (
        optuna.pruners.MedianPruner()
        if args.prune_trials
        else optuna.pruners.NopPruner()
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        load_if_exists=True,
        direction=args.direction,
        pruner=pruner,
    )

    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
            catch=(KeyboardInterrupt,),
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        print(
            "Progress has been saved. You can resume by running the same command again."
        )
        print(f"Study state saved in: {storage_url}")
        print(
            f"Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
        )
        print(
            f"Incomplete trials: {len([t for t in study.trials if t.state != optuna.trial.TrialState.COMPLETE])}"
        )

    print("\n" + "=" * 50)
    print("Optimization finished!")
    print("=" * 50)
    print(f"Number of finished trials: {len(study.trials)}")
    print(
        f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}"
    )
    print(
        f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}"
    )

    if study.best_trial is not None:
        print("\nBest trial:")
        print(f"  Value: {study.best_trial.value:.4f}")
        print("\n  Params:")
        for key, value in study.best_trial.params.items():
            print(f"    {key}: {value}")

        best_params_path = os.path.join(log_dir, "best_hyperparameters.pkl")
        with open(best_params_path, "wb") as f:
            pickle.dump(
                {
                    "best_value": study.best_trial.value,
                    "best_params": study.best_trial.params,
                    "best_trial_number": study.best_trial.number,
                },
                f,
            )
        print(f"\nBest hyperparameters saved to: {best_params_path}")

        best_trial_dir = os.path.join(log_dir, f"trial_{study.best_trial.number}")
        best_model_path = os.path.join(best_trial_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            import shutil

            shutil.copy(best_model_path, os.path.join(log_dir, "best_model.zip"))
            print(f"Best model copied to: {os.path.join(log_dir, 'best_model.zip')}")

    study_summary_path = os.path.join(log_dir, "optuna_study_summary.txt")
    with open(study_summary_path, "w") as f:
        f.write("Optuna Study Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Study name: {args.study_name}\n")
        f.write(f"Direction: {args.direction}\n")
        f.write(f"Total trials: {len(study.trials)}\n\n")

        if study.best_trial is not None:
            f.write(f"Best trial number: {study.best_trial.number}\n")
            f.write(f"Best value: {study.best_trial.value:.4f}\n\n")
            f.write("Best hyperparameters:\n")
            for key, value in study.best_trial.params.items():
                f.write(f"  {key}: {value}\n")

    print(f"\nStudy summary saved to: {study_summary_path}")


if __name__ == "__main__":
    main()
