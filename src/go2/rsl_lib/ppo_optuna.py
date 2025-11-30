import argparse
import os
import pickle

import optuna
import torch
import numpy as np
import genesis as gs

from rsl_rl.runners import OnPolicyRunner

from src.go2 import Go2Env
from src.configs import get_cfgs, set_global_seed
from src.configs.algorithm_config import (
    AlgorithmCfg,
    PolicyCfg,
    TrainCfg,
    RunnerInnerCfg,
)
from dataclasses import asdict


def evaluate_ppo_policy(
    env, runner, n_episodes: int = 10, device: str = "cuda"
) -> float:
    """
    Evaluate a PPO policy by running n_episodes and returning the mean return.
    """
    policy = runner.get_inference_policy(device=device)
    returns = []

    with torch.no_grad():
        for _ in range(n_episodes):
            obs, _ = env.reset()
            obs = obs.to(device)

            done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            ep_returns = torch.zeros(env.num_envs, device=device)

            while not done.all():
                actions = policy(obs)
                next_obs, rewards, dones, _ = env.step(actions)

                ep_returns += rewards
                done |= dones.bool()
                obs = next_obs

            returns.extend(ep_returns[done].cpu().numpy().tolist())

    return float(np.mean(returns)) if len(returns) > 0 else 0.0


def find_latest_checkpoint(trial_log_dir: str) -> tuple[str | None, int]:
    """
    Find the latest checkpoint in the trial directory.
    Returns (checkpoint_path, iteration_number) or (None, 0) if no checkpoint found.
    """
    if not os.path.exists(trial_log_dir):
        return None, 0

    # OnPolicyRunner saves checkpoints as model_{iteration}.pt
    checkpoint_files = [
        f
        for f in os.listdir(trial_log_dir)
        if f.startswith("model_") and f.endswith(".pt")
    ]

    if not checkpoint_files:
        return None, 0

    # Extract iteration numbers and find the latest
    iterations = []
    for f in checkpoint_files:
        try:
            iter_num = int(f.replace("model_", "").replace(".pt", ""))
            iterations.append((iter_num, os.path.join(trial_log_dir, f)))
        except ValueError:
            continue

    if not iterations:
        return None, 0

    iterations.sort(key=lambda x: x[0], reverse=True)
    latest_iter, latest_path = iterations[0]
    return latest_path, latest_iter


def objective(trial, args):
    """
    Optuna objective function that suggests hyperparameters, trains PPO, and returns evaluation metric.
    Supports resuming from checkpoints if a trial was interrupted.
    Note: set_global_seed() should be called once in main(), not here, to avoid reinitializing Genesis.
    """
    # Create trial-specific log directory
    trial_log_dir = os.path.join("logs", args.exp_name, f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    # Check if we can resume from a checkpoint
    checkpoint_path, checkpoint_iter = find_latest_checkpoint(trial_log_dir)
    resume_from_checkpoint = checkpoint_path is not None and args.resume_trials

    if resume_from_checkpoint:
        print(
            f"Trial {trial.number}: Resuming from checkpoint at iteration {checkpoint_iter}"
        )
        # Load configs from existing checkpoint
        try:
            with open(os.path.join(trial_log_dir, "cfgs.pkl"), "rb") as f:
                env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_dict = pickle.load(
                    f
                )
        except FileNotFoundError:
            print(f"Trial {trial.number}: Config file not found, starting from scratch")
            resume_from_checkpoint = False
    else:
        # Get environment configs
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    if not resume_from_checkpoint:
        # Suggest hyperparameters for PPO algorithm
        # Ranges adjusted to be closer to defaults that work well
        algorithm_cfg = AlgorithmCfg(
            class_name="PPO",
            clip_param=trial.suggest_float("clip_param", 0.15, 0.25, log=False),
            desired_kl=trial.suggest_float("desired_kl", 0.008, 0.015, log=True),
            entropy_coef=trial.suggest_float("entropy_coef", 1e-6, 0.02, log=True),
            gamma=trial.suggest_float("gamma", 0.98, 0.999, log=False),
            lam=trial.suggest_float("lam", 0.92, 0.98, log=False),
            learning_rate=trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
            max_grad_norm=trial.suggest_float("max_grad_norm", 0.8, 1.5, log=False),
            num_learning_epochs=trial.suggest_int("num_learning_epochs", 4, 8),
            num_mini_batches=trial.suggest_int("num_mini_batches", 4, 8),
            schedule="adaptive",
            use_clipped_value_loss=True,
            value_loss_coef=trial.suggest_float("value_loss_coef", 0.8, 1.5, log=False),
        )

        # Suggest hyperparameters for policy network
        # Using larger networks closer to defaults [512, 256, 128]
        hidden_dim_1 = trial.suggest_int("hidden_dim_1", 256, 512, step=64)
        hidden_dim_2 = trial.suggest_int("hidden_dim_2", 128, 256, step=64)
        hidden_dim_3 = trial.suggest_int("hidden_dim_3", 64, 128, step=32)

        policy_cfg = PolicyCfg(
            class_name="ActorCritic",
            activation=trial.suggest_categorical("activation", ["elu", "relu", "tanh"]),
            actor_hidden_dims=[hidden_dim_1, hidden_dim_2, hidden_dim_3],
            critic_hidden_dims=[hidden_dim_1, hidden_dim_2, hidden_dim_3],
            init_noise_std=trial.suggest_float("init_noise_std", 0.5, 2.0, log=False),
        )

        # Create runner config
        runner_cfg = RunnerInnerCfg(
            experiment_name=args.exp_name,
            max_iterations=args.max_iterations,
            log_interval=1,
        )

        # Create training config
        train_cfg = TrainCfg(
            algorithm=algorithm_cfg,
            policy=policy_cfg,
            runner=runner_cfg,
            num_steps_per_env=args.num_steps_per_env,
            save_interval=args.save_interval,
        )

        # Save configs
        train_cfg_dict = asdict(train_cfg)
        pickle.dump(
            [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg_dict],
            open(os.path.join(trial_log_dir, "cfgs.pkl"), "wb"),
        )
    else:
        # When resuming, we already loaded train_cfg_dict from the checkpoint
        # Calculate remaining iterations
        remaining_iterations = max(0, args.max_iterations - checkpoint_iter)
        if remaining_iterations == 0:
            print(f"Trial {trial.number}: Already completed, skipping...")
            # Try to evaluate if model exists
            try:
                eval_env = Go2Env(
                    num_envs=1,  # Evaluation should use 1 env
                    env_cfg=env_cfg,
                    obs_cfg=obs_cfg,
                    reward_cfg=reward_cfg,
                    command_cfg=command_cfg,
                    show_viewer=False,
                )
                runner = OnPolicyRunner(
                    eval_env, train_cfg_dict, trial_log_dir, device=gs.device
                )
                runner.load(checkpoint_path)
                mean_return = evaluate_ppo_policy(
                    eval_env, runner, args.n_eval_episodes, gs.device
                )
                trial.set_user_attr("mean_return", mean_return)
                return mean_return
            except Exception as e:
                print(f"Trial {trial.number}: Error evaluating completed trial: {e}")
                return float("-inf")

    # Create training environment
    # Note: show_viewer=False is set explicitly, but this shouldn't affect performance
    train_env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # Create evaluation environment
    eval_env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # Create runner and train
    runner = OnPolicyRunner(train_env, train_cfg_dict, trial_log_dir, device=gs.device)

    # Initialize remaining_iterations
    remaining_iterations = args.max_iterations

    # Load checkpoint if resuming
    if resume_from_checkpoint:
        try:
            runner.load(checkpoint_path)
            print(
                f"Trial {trial.number}: Loaded checkpoint from iteration {checkpoint_iter}"
            )
            # Update max_iterations to remaining iterations
            remaining_iterations = max(0, args.max_iterations - checkpoint_iter)
            if remaining_iterations == 0:
                print(f"Trial {trial.number}: Training already complete, evaluating...")
            else:
                print(
                    f"Trial {trial.number}: Continuing training for {remaining_iterations} more iterations"
                )
        except Exception as e:
            print(
                f"Trial {trial.number}: Failed to load checkpoint: {e}, starting from scratch"
            )
            resume_from_checkpoint = False
            remaining_iterations = args.max_iterations

    try:
        if resume_from_checkpoint and remaining_iterations > 0:
            # Resume training with remaining iterations
            runner.learn(
                num_learning_iterations=remaining_iterations,
                init_at_random_ep_len=False,  # Don't reinitialize when resuming
            )
        elif not resume_from_checkpoint:
            # Start fresh training
            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )

        # Evaluate the trained policy
        mean_return = evaluate_ppo_policy(
            eval_env, runner, n_episodes=args.n_eval_episodes, device=gs.device
        )

        # Report the metric to Optuna
        trial.set_user_attr("mean_return", mean_return)

        return mean_return

    except KeyboardInterrupt:
        # Handle interruption gracefully
        print(f"\nTrial {trial.number}: Interrupted by user. Checkpoint saved.")
        # The runner should have saved checkpoints automatically via save_interval
        raise  # Re-raise to allow Optuna to handle it

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        import traceback

        traceback.print_exc()
        # Return a very low value so Optuna knows this trial failed
        return float("-inf")

    finally:
        # Clean up if needed (optional - you might want to keep logs)
        if args.prune_trials:
            # Optionally remove failed or pruned trials to save space
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter optimization for PPO"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-ppo-optuna")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=500,
        help="Number of learning iterations per trial",
    )
    parser.add_argument(
        "--num_steps_per_env",
        type=int,
        default=24,
        help="Number of steps per environment per iteration (default: 24, same as ppo_train)",
    )
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes for evaluation",
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

    # Initialize Genesis once (before any trials)
    # This must be called only once, not in each trial
    set_global_seed()

    # Create main log directory
    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    # Create or load Optuna study
    if args.study_name is None:
        args.study_name = args.exp_name

    storage_url = args.storage or f"sqlite:///{os.path.join(log_dir, 'optuna.db')}"

    # Create study with pruning if enabled
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

    # Run optimization with proper exception handling
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            show_progress_bar=True,
            catch=(KeyboardInterrupt,),  # Allow graceful interruption
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

    # Print results
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

        # Save best hyperparameters
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

    # Save study summary
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
