import argparse
import os
import pickle

import torch

from src.go2 import Go2Env
from src.algorithms.sac import SACAgent, SACConfig
from src.algorithms.off_policy_runner import OffPolicyRunner
from src.configs.seed import set_global_seed
from src.algorithms.early_stopping import evaluate_agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-custom-sac")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Caminho para o checkpoint (.pt). Se None, usa best_model.pt do exp.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    parser.add_argument("--eval_num_envs", type=int, default=1)
    parser.add_argument("--show_viewer", action="store_true")
    args = parser.parse_args()

    set_global_seed()

    log_dir = os.path.join("logs", args.exp_name)

    cfg_path = os.path.join(log_dir, "cfgs_custom.pkl")
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)

    eval_env = Go2Env(
        num_envs=args.eval_num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.show_viewer,
    )

    sac_cfg = SACConfig(
        obs_dim=eval_env.num_obs,
        act_dim=eval_env.num_actions,
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

    if args.checkpoint is not None:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(log_dir, "best_model.pt")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint não encontrado em: {ckpt_path}")

    print(f"Carregando checkpoint de: {ckpt_path}")
    OffPolicyRunner.load_checkpoint(agent, ckpt_path, device=args.device)

    mean_return = evaluate_agent(
        eval_env,
        agent,
        n_episodes=args.n_eval_episodes,
        device=torch.device(args.device),
    )
    print(
        f"Retorno médio em {args.n_eval_episodes} episódios de avaliação: {mean_return:.3f}"
    )

    if args.show_viewer:
        print("Rodando mais 3 episódios com viewer para inspeção visual...")
        for ep in range(3):
            obs, _ = eval_env.reset()
            obs = obs.to(args.device)

            done = torch.zeros(eval_env.num_envs, dtype=torch.bool, device=args.device)
            ep_ret = torch.zeros(eval_env.num_envs, device=args.device)

            while not done.all():
                with torch.no_grad():
                    actions = agent.act(obs, eval_mode=True)
                    next_obs, rewards, dones, _ = eval_env.step(actions)

                ep_ret += rewards.to(args.device)
                done |= dones.to(args.device).bool()
                obs = next_obs.to(args.device)

            print(
                f"[viewer] Episódio {ep + 1}: retorno médio por env = {ep_ret.mean().item():.3f}"
            )


if __name__ == "__main__":
    main()
