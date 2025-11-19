import os
import pickle
import shutil
from argparse import ArgumentParser

import genesis as gs
from rsl_rl.runners import OnPolicyRunner

from src.configs import get_cfgs, get_train_cfg
from src.go2 import Go2Env


def main():
    parser = ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=10000)
    parser.add_argument("--device", type=str, default="cuda:0", help="device to use: 'cpu' or 'cuda:0'")
    args = parser.parse_args()

    backend = gs.constants.backend.gpu if args.device.lower() == "cuda:0" else gs.constants.backend.cpu
    gs.init(logging_level="warning", backend=backend)

    log_dir  = f"./logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    os.makedirs(log_dir, exist_ok=True)

    env = Go2Env(
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        num_envs=args.num_envs,
        device=args.device,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=args.device)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/configs.pkl", "wb"),
    )

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

if __name__ == "__main__":
    main()