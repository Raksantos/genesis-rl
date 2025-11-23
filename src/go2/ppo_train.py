import argparse
import os
import pickle
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from src.go2 import Go2Env
from src.configs import get_cfgs, get_train_cfg
from src.algorithms import OffPolicyRunner, OffPolicyRunnerConfig, SACAgent, SACConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("-a", "--algorithm", type=str, default="ppo")
    args = parser.parse_args()

    gs.init(backend=gs.constants.backend.gpu, logging_level="Warning")

    log_dir = f"logs/{args.exp_name + '-' + args.algorithm}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
    )

    match args.algorithm.lower():
        case "ppo":
            runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

            runner.learn(
                num_learning_iterations=args.max_iterations, init_at_random_ep_len=True
            )
        case "sac":
            agent = SACAgent(
                SACConfig(
                    obs_dim=env.num_obs,
                    act_dim=env.num_actions,
                    device=gs.device,
                    action_scale=env_cfg["action_scale"],
                )
            )

            runner_cfg = OffPolicyRunnerConfig(
                total_env_steps=args.max_iterations
                * args.num_envs
                * train_cfg["num_steps_per_env"],
                init_random_steps=10_000,
                batch_size=256,
            )

            runner = OffPolicyRunner(
                env=env,
                agent=agent,
                runner_cfg=runner_cfg,
                buffer_size=1_000_000,
                log_dir=log_dir,
            )

            runner.run()
        case _:
            raise ValueError(f"Algorithm {args.algorithm} not recognized.")


if __name__ == "__main__":
    main()
