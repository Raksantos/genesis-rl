# ddpg_custom_train.py
import time
import numpy as np
import torch


def evaluate_ddpg_policy(
    env,
    agent,
    eval_episodes: int = 5,
    device: str = "cpu",
) -> float:
    """
    Roda alguns episódios em modo avaliação (sem ruído de exploração)
    e retorna a recompensa média.
    """
    agent.actor.eval()
    returns = []

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_return = 0.0

        while not (done or truncated):
            with torch.no_grad():
                # sem ruído na avaliação
                action = agent.select_action(obs, noise_scale=0.0)[0]
            obs, rew, done, truncated, info = env.step(action)
            episode_return += rew

        returns.append(episode_return)

    agent.actor.train()
    return float(np.mean(returns))


def ddpg_custom_train(
    env,
    eval_env,
    agent,
    replay_buffer,
    total_timesteps: int = 1_000_000,
    start_steps: int = 10_000,
    batch_size: int = 256,
    update_after: int = 10_000,
    update_every: int = 50,
    eval_interval: int = 10_000,
    eval_episodes: int = 5,
    max_episode_steps: int | None = None,
    log_callback=None,
    device: str = "cuda",
):
    """
    Loop de treinamento "puro" DDPG, análogo ao SAC/TD3 custom train.

    Parâmetros principais:
      - env: ambiente de treinamento Gym-like (single env)
      - eval_env: ambiente de avaliação
      - agent: instância de DDPG
      - replay_buffer: instância de ReplayBuffer
      - total_timesteps: número total de steps de interação com o env
      - start_steps: quantos steps usar ações aleatórias antes de usar a policy
      - batch_size: tamanho do batch nas atualizações
      - update_after: só começa a treinar depois desse número de steps
      - update_every: quantos env steps entre blocos de updates
      - eval_interval: a cada quantos steps fazer avaliação
      - eval_episodes: quantos episódios para cada avaliação
      - max_episode_steps: se quiser forçar um limite de steps/episódio
      - log_callback: função opcional para logar métricas (dict) ao longo do treino
    """
    device = device or agent.device
    obs, _ = env.reset()
    ep_return = 0.0
    ep_len = 0
    global_step = 0

    start_time = time.time()

    while global_step < total_timesteps:
        # ----------------------------------------------------------
        # Escolha de ação (exploração vs política)
        # ----------------------------------------------------------
        if global_step < start_steps:
            action = env.action_space.sample()
        else:
            # DDPG: normalmente usa ruído aditivo gaussiano/OU
            action = agent.select_action(obs, noise_scale=0.1)[0]

        next_obs, reward, done, truncated, info = env.step(action)
        ep_return += reward
        ep_len += 1
        global_step += 1

        # gerenciamento de fim de episódio
        done_env = done or truncated
        if max_episode_steps is not None and ep_len >= max_episode_steps:
            done_env = True

        replay_buffer.add(obs, action, reward, next_obs, done_env)

        obs = next_obs

        if done_env:
            # log de episódio (opcional)
            if log_callback is not None:
                log_callback(
                    {
                        "episode_return": ep_return,
                        "episode_length": ep_len,
                        "global_step": global_step,
                        "type": "train_episode",
                    }
                )

            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

        # ----------------------------------------------------------
        # Atualizações do DDPG
        # ----------------------------------------------------------
        if global_step >= update_after and global_step % update_every == 0:
            # número de gradient steps proporcional ao update_every
            for _ in range(update_every):
                info_train = agent.train_step(replay_buffer, batch_size=batch_size)
                if log_callback is not None:
                    info_train_with_step = {
                        **info_train,
                        "global_step": global_step,
                        "type": "train_update",
                    }
                    log_callback(info_train_with_step)

        # ----------------------------------------------------------
        # Avaliação periódica
        # ----------------------------------------------------------
        if (global_step % eval_interval == 0) or (global_step == total_timesteps):
            eval_return = evaluate_ddpg_policy(
                eval_env,
                agent,
                eval_episodes=eval_episodes,
                device=device,
            )
            elapsed = time.time() - start_time

            if log_callback is not None:
                log_callback(
                    {
                        "eval_return": eval_return,
                        "global_step": global_step,
                        "elapsed_time_s": elapsed,
                        "type": "eval",
                    }
                )
            else:
                print(
                    f"[DDPG] Step {global_step}/{total_timesteps} | "
                    f"EvalReturn: {eval_return:.3f} | "
                    f"Elapsed: {elapsed:.1f}s"
                )

    return agent
