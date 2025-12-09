from dataclasses import dataclass
import os

import numpy as np
import torch
from tqdm import trange

from src.algorithms.sac import SACAgent
from src.algorithms.td3 import TD3Agent
from src.algorithms.replay_buffer import ReplayBuffer
from src.algorithms.early_stopping import EarlyStoppingConfig, evaluate_agent


@dataclass
class OffPolicyRunnerConfig:
    total_env_steps: int = 1_000_000
    init_random_steps: int = 10_000
    batch_size: int = 256
    update_every: int = 1
    updates_per_step: float = 1.0
    log_interval: int = 1000
    checkpoint_interval: int = 50_000


class OffPolicyRunner:
    def __init__(
        self,
        env,
        agent: SACAgent | TD3Agent,
        runner_cfg: OffPolicyRunnerConfig,
        buffer_size: int = 1_000_000,
        log_dir: str | None = None,
        eval_env=None,
        early_cfg: EarlyStoppingConfig | None = None,
    ):
        """
        env: ambiente de treino (vetorizado: env.num_envs, num_obs, num_actions)
        agent: instância de SACAgent ou TD3Agent
        runner_cfg: hiperparâmetros do loop off-policy
        buffer_size: tamanho do replay buffer
        log_dir: diretório para salvar checkpoints e logs
        eval_env: ambiente separado para avaliação (pode ter num_envs diferente)
        early_cfg: config de avaliação periódica + early stopping
        """
        self.env = env
        self.agent = agent
        self.cfg = runner_cfg
        self.log_dir = log_dir

        self.device = torch.device(agent.config.device)

        self.num_envs = env.num_envs
        obs_dim = env.num_obs
        act_dim = env.num_actions

        self.replay_buffer = ReplayBuffer(
            obs_dim=obs_dim,
            act_dim=act_dim,
            size=buffer_size,
            device=str(self.device),
        )

        self.episode_rewards = np.zeros(self.num_envs, dtype=np.float32)
        self.completed_returns: list[float] = []

        self.total_return_sum: float = 0.0
        self.total_episodes: int = 0

        self.eval_env = eval_env
        self.early_cfg = early_cfg
        self.best_eval_return: float | None = None
        self.no_improvement_evals: int = 0
        self.num_evals: int = 0

        self.global_step: int = 0

    def run(self):
        obs, _ = self.env.reset()
        obs = obs.to(self.device)

        self.global_step = 0
        logs: dict[str, float] = {}

        iters = self.cfg.total_env_steps // self.num_envs
        pbar = trange(iters)

        for _ in pbar:
            if self.global_step < self.cfg.init_random_steps:
                actions = (
                    2
                    * torch.rand(
                        (self.num_envs, self.env.num_actions), device=self.device
                    )
                    - 1
                )
                actions = actions * self.agent.config.action_scale
            else:
                with torch.no_grad():
                    actions = self.agent.act(obs, eval_mode=False)

            with torch.no_grad():
                next_obs, rewards, dones, _ = self.env.step(actions)

            next_obs = next_obs.to(self.device)
            rewards = rewards.unsqueeze(-1).to(self.device)
            dones = dones.unsqueeze(-1).to(self.device).float()

            self.episode_rewards += rewards.squeeze(-1).cpu().numpy()

            self.replay_buffer.store_batch(
                obs.cpu().numpy(),
                actions.cpu().numpy(),
                rewards.cpu().numpy(),
                next_obs.cpu().numpy(),
                dones.cpu().numpy(),
            )

            self.global_step += self.num_envs
            obs = next_obs

            done_envs = dones.squeeze(-1) > 0.5
            if done_envs.any():
                idx = done_envs.nonzero(as_tuple=False).squeeze(-1)

                for i in idx.cpu().numpy():
                    ep_ret = float(self.episode_rewards[i])
                    self.completed_returns.append(ep_ret)

                    self.total_return_sum += ep_ret
                    self.total_episodes += 1

                    self.episode_rewards[i] = 0.0

                self.env.reset_idx(idx)
                obs_reset, _ = self.env.get_observations()
                obs = obs_reset.to(self.device)

            if (
                self.replay_buffer.size >= self.cfg.batch_size
                and self.global_step >= self.cfg.init_random_steps
            ):
                if self.global_step % self.cfg.update_every == 0:
                    num_updates = int(self.cfg.update_every * self.cfg.updates_per_step)
                    num_updates = max(num_updates, 1)

                    for _ in range(num_updates):
                        batch = self.replay_buffer.sample_batch(self.cfg.batch_size)
                        logs = self.agent.update(batch)

            if (
                self.eval_env is not None
                and self.early_cfg is not None
                and self.global_step > 0
                and self.global_step % self.early_cfg.eval_interval_steps == 0
            ):
                eval_return = evaluate_agent(
                    self.eval_env,
                    self.agent,
                    self.early_cfg.n_eval_episodes,
                    self.device,
                )
                self.num_evals += 1

                if self.log_dir is not None:
                    eval_log_dir = os.path.join(self.log_dir, "eval_logs")
                    os.makedirs(eval_log_dir, exist_ok=True)
                    with open(os.path.join(eval_log_dir, "eval_returns.txt"), "a") as f:
                        f.write(f"{self.global_step},{eval_return}\n")

                improved = (
                    self.best_eval_return is None or eval_return > self.best_eval_return
                )

                if improved:
                    self.best_eval_return = eval_return
                    self.no_improvement_evals = 0

                    if self.log_dir is not None:
                        best_path = os.path.join(self.log_dir, "best_model.pt")
                    else:
                        best_path = "best_model.pt"

                    self.save_checkpoint(best_path, self.global_step)

                else:
                    self.no_improvement_evals += 1

                if (
                    self.num_evals >= self.early_cfg.min_evals
                    and self.no_improvement_evals
                    >= self.early_cfg.max_no_improvement_evals
                ):
                    print(
                        f"[EarlyStopping] Parando treinamento em step={self.global_step} "
                        f"sem melhoria em {self.no_improvement_evals} avaliações. "
                        f"Melhor retorno avaliado = {self.best_eval_return:.2f}"
                    )

                    model_name = (
                        "td3_final.pt"
                        if isinstance(self.agent, TD3Agent)
                        else "sac_final.pt"
                    )
                    final_path = (
                        os.path.join(self.log_dir, model_name)
                        if self.log_dir is not None
                        else model_name
                    )
                    self.save_checkpoint(final_path, self.global_step)
                    return

            if (
                self.log_dir is not None
                and self.global_step > 0
                and self.global_step % self.cfg.checkpoint_interval == 0
            ):
                ckpt_path = os.path.join(
                    self.log_dir, f"checkpoint_{self.global_step}.pt"
                )
                self.save_checkpoint(ckpt_path, self.global_step)

            if self.global_step % self.cfg.log_interval == 0 and logs:
                if self.completed_returns:
                    avg_last_10 = float(np.mean(self.completed_returns[-10:]))
                else:
                    avg_last_10 = 0.0

                if self.total_episodes > 0:
                    avg_all = self.total_return_sum / self.total_episodes
                else:
                    avg_all = 0.0

                pbar.set_description(
                    f"🔁 step={self.global_step} | "
                    f"🎯 avg10={avg_last_10:.2f} | "
                    f"📉 mean={avg_all:.2f} | "
                    f"🔥 α={logs.get('alpha', 0):.3f} | "
                    f"🎭 actor={logs.get('actor_loss', 0):.3f} | "
                    f"🧠 critic={logs.get('critic_loss', 0):.3f}"
                )

        model_name = (
            "td3_final.pt" if isinstance(self.agent, TD3Agent) else "sac_final.pt"
        )
        final_path = (
            os.path.join(self.log_dir, model_name)
            if self.log_dir is not None
            else model_name
        )
        self.save_checkpoint(final_path, self.global_step)

    def save_checkpoint(self, path: str, global_step: int):
        checkpoint = {
            "global_step": global_step,
            "actor": self.agent.actor.state_dict(),
            "q1": self.agent.q1.state_dict(),
            "q2": self.agent.q2.state_dict(),
            "q1_target": self.agent.q1_target.state_dict(),
            "q2_target": self.agent.q2_target.state_dict(),
            "actor_optimizer": self.agent.actor_optimizer.state_dict(),
            "q1_optimizer": self.agent.q1_optimizer.state_dict(),
            "q2_optimizer": self.agent.q2_optimizer.state_dict(),
        }

        if hasattr(self.agent, "alpha_opt") and hasattr(self.agent, "log_alpha"):
            checkpoint["alpha_opt"] = self.agent.alpha_opt.state_dict()
            checkpoint["log_alpha"] = self.agent.log_alpha.detach().cpu()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(agent: SACAgent | TD3Agent, path: str, device="cuda"):
        ckpt = torch.load(path, map_location=device)

        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_target.load_state_dict(ckpt["q1_target"])
        agent.q2_target.load_state_dict(ckpt["q2_target"])

        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent.q1_optimizer.load_state_dict(ckpt["q1_optimizer"])
        agent.q2_optimizer.load_state_dict(ckpt["q2_optimizer"])

        if hasattr(agent, "alpha_opt") and "alpha_opt" in ckpt:
            agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])
        if hasattr(agent, "log_alpha") and "log_alpha" in ckpt:
            agent.log_alpha = ckpt["log_alpha"].to(device).requires_grad_(True)

        return ckpt["global_step"]
