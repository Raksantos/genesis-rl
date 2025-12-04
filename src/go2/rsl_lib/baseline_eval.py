import argparse
import os
import sys
import pickle
from pathlib import Path
from collections import deque

import numpy as np
import torch
import genesis as gs

# Adiciona o diretório raiz do projeto ao path quando executado diretamente
if __name__ == "__main__":
    # Quando executado como script, adiciona o diretório raiz ao path
    script_path = Path(__file__).resolve()
    # Vai 4 níveis acima: baseline_eval.py -> rsl_lib -> go2 -> src -> raiz
    project_root = script_path.parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.go2 import Go2Env
from src.configs import set_global_seed
from .baseline_train import RandomBaseline


def evaluate_baseline(
    env: Go2Env,
    baseline: RandomBaseline,
    n_episodes: int = 10,
    device: str = "cuda",
):
    """
    Avalia o baseline por n_episodes e retorna métricas.

    Args:
        env: Ambiente de simulação
        baseline: Política baseline
        n_episodes: Número de episódios para avaliação
        device: Dispositivo (cuda/cpu)

    Returns:
        Dict com métricas de avaliação
    """
    episode_rewards = []
    episode_lengths = []

    obs, _ = env.reset()
    obs = obs.to(device)

    current_reward = 0.0
    current_length = 0
    episodes_completed = 0

    print(f"Avaliando baseline por {n_episodes} episódios...")

    with torch.no_grad():
        while episodes_completed < n_episodes:
            # Ação aleatória
            actions = baseline.act(obs)

            # Step no ambiente
            obs, rewards, dones, extras = env.step(actions)
            obs = obs.to(device)
            rewards = rewards.to(device)
            dones = dones.to(device)

            current_reward += float(rewards[0].item())
            current_length += 1

            # Detecta fim de episódio
            if dones[0].item() > 0.5:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                episodes_completed += 1

                print(
                    f"Episódio {episodes_completed}/{n_episodes}: "
                    f"Recompensa = {current_reward:.3f}, "
                    f"Comprimento = {current_length}"
                )

                # Reset
                obs, _ = env.reset()
                obs = obs.to(device)
                current_reward = 0.0
                current_length = 0

    metrics = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Avalia baseline com política aleatória"
    )
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking-baseline")
    parser.add_argument("--n_episodes", type=int, default=10)
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"]
    )
    args = parser.parse_args()

    set_global_seed()

    log_dir = f"logs/{args.exp_name}"

    # Carrega configurações
    if os.path.exists(os.path.join(log_dir, "cfgs.pkl")):
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(
            open(os.path.join(log_dir, "cfgs.pkl"), "rb")
        )
    else:
        # Se não encontrar, usa configurações padrão
        from src.configs import get_cfgs, get_train_cfg

        env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
        train_cfg = get_train_cfg(args.exp_name, 1000)

    # Desabilita recompensas para avaliação (opcional)
    reward_cfg["reward_scales"] = {}

    # Ajusta terminações para permitir episódios mais longos
    env_cfg["termination_if_roll_greater_than"] = 50
    env_cfg["termination_if_pitch_greater_than"] = 50

    # Cria ambiente (1 ambiente para avaliação)
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Cria baseline
    # O ambiente espera ações no range [-clip_actions, clip_actions]
    # e depois aplica action_scale internamente
    clip_actions = env_cfg.get("clip_actions", 1.0)
    baseline = RandomBaseline(
        num_actions=env.num_actions,
        clip_actions=clip_actions,
        device=args.device,
    )

    # Avalia
    metrics = evaluate_baseline(
        env, baseline, n_episodes=args.n_episodes, device=args.device
    )

    # Imprime resultados
    print("\n" + "=" * 50)
    print("Resultados da Avaliação do Baseline")
    print("=" * 50)
    print(f"Episódios avaliados: {args.n_episodes}")
    print(f"\nRecompensas:")
    print(f"  Média: {metrics['mean_reward']:.3f}")
    print(f"  Desvio padrão: {metrics['std_reward']:.3f}")
    print(f"  Mínimo: {np.min(metrics['episode_rewards']):.3f}")
    print(f"  Máximo: {np.max(metrics['episode_rewards']):.3f}")
    print(f"  Mediana: {np.median(metrics['episode_rewards']):.3f}")
    print(f"\nComprimentos de Episódio:")
    print(f"  Média: {metrics['mean_length']:.1f}")
    print(f"  Desvio padrão: {metrics['std_length']:.1f}")
    print(f"  Mínimo: {np.min(metrics['episode_lengths']):.0f}")
    print(f"  Máximo: {np.max(metrics['episode_lengths']):.0f}")
    print(f"  Mediana: {np.median(metrics['episode_lengths']):.1f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
