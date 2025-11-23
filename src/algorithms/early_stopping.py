from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class EarlyStoppingConfig:
    eval_interval_steps: int = 10_000
    n_eval_episodes: int = 5
    max_no_improvement_evals: int = 10
    min_evals: int = 10
    deterministic_eval: bool = True


@torch.no_grad()
def evaluate_agent(env, agent, n_episodes: int, device: torch.device) -> float:
    """
    Roda n_episodes no eval_env e retorna o retorno médio.
    Supondo que env.num_envs >= 1 e que step/reset já são vetorizados.
    """
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        obs = obs.to(device)

        done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        ep_returns = torch.zeros(env.num_envs, device=device)

        while not done.all():
            actions = agent.act(obs, eval_mode=True)
            next_obs, rewards, dones, _ = env.step(actions)

            ep_returns += rewards
            done |= dones.bool()
            obs = next_obs

        returns.extend(ep_returns[done].cpu().numpy().tolist())

    return float(np.mean(returns)) if len(returns) > 0 else 0.0
