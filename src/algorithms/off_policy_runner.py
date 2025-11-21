from dataclasses import dataclass
import os

import numpy as np
import torch
from tqdm import trange

from src.algorithms import SACAgent
from src.algorithms.replay_buffer import ReplayBuffer


@dataclass
class OffPolicyRunnerConfig:
    total_env_steps: int = 1_000_000
    init_random_steps: int = 10_000
    batch_size: int = 256
    update_every: int = 1
    updates_per_step: float = 1.0
    log_interval: int = 1000
    eval_interval: int = 50_000


class OffPolicyRunner:
    def __init__(
        self,
        env,
        agent: SACAgent,
        runner_cfg: OffPolicyRunnerConfig,
        buffer_size: int = 1_000_000,
        log_dir: str | None = None,
    ):
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
        self.completed_returns = []

    def run(self):
        obs, _ = self.env.reset()
        obs = obs.to(self.device)

        global_step = 0
        logs: dict[str, float] = {}

        pbar = trange(self.cfg.total_env_steps // self.num_envs)
        for _ in pbar:
            if global_step < self.cfg.init_random_steps:
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

            global_step += self.num_envs
            obs = next_obs

            done_envs = dones.squeeze(-1) > 0.5
            if done_envs.any():
                idx = done_envs.nonzero(as_tuple=False).squeeze(-1)

                for i in idx.cpu().numpy():
                    ep_ret = float(self.episode_rewards[i])
                    self.completed_returns.append(ep_ret)
                    self.episode_rewards[i] = 0.0

                self.env.reset_idx(idx)
                obs_reset, _ = self.env.get_observations()
                obs = obs_reset.to(self.device)

            if (
                self.replay_buffer.size >= self.cfg.batch_size
                and global_step >= self.cfg.init_random_steps
            ):
                if global_step % self.cfg.update_every == 0:
                    batch = self.replay_buffer.sample_batch(self.cfg.batch_size)
                    logs = self.agent.update(batch)

            if global_step > 0 and global_step % self.cfg.eval_interval == 0:
                self.save_checkpoint(
                    path=os.path.join(
                        self.log_dir or ".", f"checkpoint_{global_step}.pt"
                    ),
                    global_step=global_step,
                )

            if global_step % self.cfg.log_interval == 0 and logs:
                if self.completed_returns:
                    avg_ret = float(np.mean(self.completed_returns[-10:]))
                else:
                    avg_ret = 0.0

                pbar.set_description(
                    f"step={global_step} "
                    f"R_ep(avg10)={avg_ret:.2f} "
                    f"alpha={logs.get('alpha', 0):.3f} "
                    f"actor_loss={logs.get('actor_loss', 0):.3f} "
                    f"critic_loss={logs.get('critic_loss', 0):.3f}"
                )

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
            "alpha_opt": self.agent.alpha_opt.state_dict(),
            "log_alpha": self.agent.log_alpha.detach().cpu(),
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    @staticmethod
    def load_checkpoint(agent: SACAgent, path: str, device="cuda"):
        ckpt = torch.load(path, map_location=device)

        agent.actor.load_state_dict(ckpt["actor"])
        agent.q1.load_state_dict(ckpt["q1"])
        agent.q2.load_state_dict(ckpt["q2"])
        agent.q1_target.load_state_dict(ckpt["q1_target"])
        agent.q2_target.load_state_dict(ckpt["q2_target"])

        agent.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        agent.q1_optimizer.load_state_dict(ckpt["q1_optimizer"])
        agent.q2_optimizer.load_state_dict(ckpt["q2_optimizer"])
        agent.alpha_opt.load_state_dict(ckpt["alpha_opt"])

        agent.log_alpha = ckpt["log_alpha"].to(device).requires_grad_(True)

        return ckpt["global_step"]
