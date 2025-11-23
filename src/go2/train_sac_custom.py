# src/go2/train_sac_custom.py

import argparse
import os
import pickle

from src.go2 import Go2Env
from src.configs import get_cfgs
from src.configs.seed import set_global_seed

from src.algorithms.sac import SACAgent, SACConfig
from src.algorithms.off_policy_runner import (
    OffPolicyRunner,
    OffPolicyRunnerConfig,
    EarlyStoppingConfig,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-custom-sac")
    parser.add_argument("--total_timesteps", type=int, default=20_000_000)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--buffer_size", type=int, default=1_000_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--init_random_steps", type=int, default=10_000)
    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--updates_per_step", type=float, default=1.0)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--n_eval_episodes", type=int, default=5)
    parser.add_argument("--train_num_envs", type=int, default=1024)
    parser.add_argument("--eval_num_envs", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=50_000)
    args = parser.parse_args()

    set_global_seed()

    log_dir = os.path.join("logs", args.exp_name)
    os.makedirs(log_dir, exist_ok=True)

    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()[:4]

    # salva configs para reprodutibilidade
    with open(os.path.join(log_dir, "cfgs_custom.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg], f)

    # ------------- ENV DE TREINO (Go2Env vetorizado) -------------
    train_env = Go2Env(
        num_envs=args.train_num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    # ------------- ENV DE AVALIAÇÃO (também Go2Env) -------------
    eval_env = Go2Env(
        num_envs=args.eval_num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,  # pode ligar True depois para ver o robô
    )

    # ------------- SAC CONFIG / AGENT -------------
    sac_cfg = SACConfig(
        obs_dim=train_env.num_obs,
        act_dim=train_env.num_actions,
        hidden_dims=[256, 256],
        gamma=0.99,
        tau=0.005,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        device=args.device,
        action_scale=1.0,
    )
    agent = SACAgent(sac_cfg)

    # ------------- RUNNER CONFIG -------------
    runner_cfg = OffPolicyRunnerConfig(
        total_env_steps=args.total_timesteps,
        init_random_steps=args.init_random_steps,
        batch_size=args.batch_size,
        update_every=args.update_every,
        updates_per_step=args.updates_per_step,
        log_interval=1_000,
        checkpoint_interval=args.checkpoint_interval,
    )

    early_cfg = EarlyStoppingConfig(
        eval_interval_steps=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        max_no_improvement_evals=10,
        min_evals=10,
        deterministic_eval=True,
    )

    runner = OffPolicyRunner(
        env=train_env,
        agent=agent,
        runner_cfg=runner_cfg,
        buffer_size=args.buffer_size,
        log_dir=log_dir,
        eval_env=eval_env,
        early_cfg=early_cfg,
    )

    runner.run()


if __name__ == "__main__":
    main()
