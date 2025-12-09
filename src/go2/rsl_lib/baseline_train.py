import argparse
import os
import sys
import pickle
import shutil
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import genesis as gs


if __name__ == "__main__":
    script_path = Path(__file__).resolve()

    project_root = script_path.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.go2 import Go2Env
from src.configs import get_cfgs, get_train_cfg, set_global_seed


class RandomBaseline:
    """
    Baseline simples com política aleatória.
    Gera ações aleatórias dentro do range permitido pelo ambiente.
    O ambiente espera ações no range [-clip_actions, clip_actions] e depois
    aplica action_scale internamente.
    """

    def __init__(
        self, num_actions: int, clip_actions: float = 1.0, device: str = "cuda"
    ):
        self.num_actions = num_actions
        self.clip_actions = clip_actions
        self.device = device

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Gera ações aleatórias uniformemente distribuídas no range [-clip_actions, clip_actions].
        O ambiente fará clip e depois aplicará action_scale internamente.

        Args:
            obs: Observações do ambiente (não usadas, mas mantém interface compatível)

        Returns:
            Ações aleatórias no formato (num_envs, num_actions)
        """
        num_envs = obs.shape[0] if obs.dim() > 1 else 1

        actions = (
            2 * torch.rand((num_envs, self.num_actions), device=self.device) - 1
        ) * self.clip_actions
        return actions


def run_baseline(
    env: Go2Env,
    baseline: RandomBaseline,
    num_iterations: int,
    num_steps_per_env: int,
    log_dir: str,
    log_interval: int = 10,
):
    """
    Executa o baseline e coleta métricas.

    Args:
        env: Ambiente de simulação
        baseline: Política baseline (aleatória)
        num_iterations: Número de iterações de treinamento
        num_steps_per_env: Número de steps por ambiente por iteração
        log_dir: Diretório para salvar logs
        log_interval: Intervalo para logging
    """
    device = gs.device
    num_envs = env.num_envs

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.int)

    all_episode_rewards = []
    all_episode_lengths = []

    iteration_rewards = []
    iteration_lengths = []

    episodes_before_iteration = 0

    obs, _ = env.reset()
    obs = obs.to(device)

    total_steps = 0

    print(f"Iniciando baseline com {num_envs} ambientes paralelos")
    print(f"Total de iterações: {num_iterations}")
    print(f"Steps por iteração: {num_steps_per_env}")

    for iteration in tqdm(range(num_iterations), desc="Iterações"):
        iteration_reward_sum = 0.0
        iteration_length_sum = 0

        for step in range(num_steps_per_env):
            actions = baseline.act(obs)

            obs, rewards, dones, extras = env.step(actions)
            obs = obs.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            episode_rewards += rewards
            episode_lengths += 1
            total_steps += num_envs

            done_envs = dones.bool()
            if done_envs.any():
                done_indices = done_envs.nonzero(as_tuple=False).squeeze(-1)

                for idx in done_indices.cpu().numpy():
                    ep_rew = float(episode_rewards[idx].item())
                    ep_len = int(episode_lengths[idx].item())

                    all_episode_rewards.append(ep_rew)
                    all_episode_lengths.append(ep_len)

                    iteration_reward_sum += ep_rew
                    iteration_length_sum += ep_len

                    episode_rewards[idx] = 0.0
                    episode_lengths[idx] = 0

            if done_envs.any():
                env.reset_idx(done_envs.nonzero(as_tuple=False).squeeze(-1))
                obs_reset, _ = env.get_observations()
                obs = obs_reset.to(device)

        num_completed_episodes = len(all_episode_rewards) - episodes_before_iteration

        if num_completed_episodes > 0:
            recent_rewards = all_episode_rewards[-num_completed_episodes:]
            recent_lengths = all_episode_lengths[-num_completed_episodes:]

            mean_reward = np.mean(recent_rewards)
            mean_length = np.mean(recent_lengths)

            iteration_rewards.append(mean_reward)
            iteration_lengths.append(mean_length)
        else:
            mean_reward = float(episode_rewards.mean().item())
            mean_length = float(episode_lengths.mean().item())
            iteration_rewards.append(mean_reward)
            iteration_lengths.append(mean_length)

        episodes_before_iteration = len(all_episode_rewards)

        if (iteration + 1) % log_interval == 0 or iteration == 0:
            total_episodes = len(all_episode_rewards)
            if total_episodes > 0:
                overall_mean_reward = np.mean(all_episode_rewards)
                overall_mean_length = np.mean(all_episode_lengths)
                overall_std_reward = np.std(all_episode_rewards)
            else:
                overall_mean_reward = mean_reward
                overall_mean_length = mean_length
                overall_std_reward = 0.0

            print(
                f"\nIteração {iteration + 1}/{num_iterations}:"
                f"\n  Episódios completados: {total_episodes}"
                f"\n  Recompensa média (geral): {overall_mean_reward:.3f} ± {overall_std_reward:.3f}"
                f"\n  Recompensa média (iteração): {mean_reward:.3f}"
                f"\n  Comprimento médio (geral): {overall_mean_length:.1f}"
                f"\n  Comprimento médio (iteração): {mean_length:.1f}"
                f"\n  Total de steps: {total_steps}"
            )

    metrics = {
        "episode_rewards": all_episode_rewards,
        "episode_lengths": all_episode_lengths,
        "iteration_rewards": iteration_rewards,
        "iteration_lengths": iteration_lengths,
        "total_steps": total_steps,
        "total_episodes": len(all_episode_rewards),
    }

    metrics_path = os.path.join(log_dir, "baseline_metrics.pkl")
    with open(metrics_path, "wb") as f:
        pickle.dump(metrics, f)

    summary_path = os.path.join(log_dir, "baseline_summary.txt")
    with open(summary_path, "w") as f:
        f.write("=== Baseline Random Policy - Resumo ===\n\n")
        f.write(f"Total de iterações: {num_iterations}\n")
        f.write(f"Steps por iteração: {num_steps_per_env}\n")
        f.write(f"Total de steps: {total_steps}\n")
        f.write(f"Total de episódios: {len(all_episode_rewards)}\n\n")

        if len(all_episode_rewards) > 0:
            f.write("Métricas de Recompensa:\n")
            f.write(f"  Média: {np.mean(all_episode_rewards):.3f}\n")
            f.write(f"  Desvio padrão: {np.std(all_episode_rewards):.3f}\n")
            f.write(f"  Mínimo: {np.min(all_episode_rewards):.3f}\n")
            f.write(f"  Máximo: {np.max(all_episode_rewards):.3f}\n")
            f.write(f"  Mediana: {np.median(all_episode_rewards):.3f}\n\n")

            f.write("Métricas de Comprimento de Episódio:\n")
            f.write(f"  Média: {np.mean(all_episode_lengths):.1f}\n")
            f.write(f"  Desvio padrão: {np.std(all_episode_lengths):.1f}\n")
            f.write(f"  Mínimo: {np.min(all_episode_lengths):.0f}\n")
            f.write(f"  Máximo: {np.max(all_episode_lengths):.0f}\n")
            f.write(f"  Mediana: {np.median(all_episode_lengths):.1f}\n")

    print(f"\n=== Baseline Finalizado ===")
    print(f"Métricas salvas em: {metrics_path}")
    print(f"Resumo salvo em: {summary_path}")

    if len(all_episode_rewards) > 0:
        print(
            f"\nRecompensa média final: {np.mean(all_episode_rewards):.3f} ± {np.std(all_episode_rewards):.3f}"
        )
        print(
            f"Comprimento médio final: {np.mean(all_episode_lengths):.1f} ± {np.std(all_episode_lengths):.1f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Treina baseline com política aleatória"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=2048)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--num_steps_per_env", type=int, default=24)
    parser.add_argument("--log_interval", type=int, default=10)
    args = parser.parse_args()

    set_global_seed()

    log_dir = f"logs/{args.exp_name}-baseline"
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

    clip_actions = env_cfg.get("clip_actions", 1.0)
    baseline = RandomBaseline(
        num_actions=env.num_actions,
        clip_actions=clip_actions,
        device=gs.device,
    )

    run_baseline(
        env=env,
        baseline=baseline,
        num_iterations=args.max_iterations,
        num_steps_per_env=args.num_steps_per_env,
        log_dir=log_dir,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    main()
