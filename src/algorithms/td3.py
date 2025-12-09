import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.algorithms.replay_buffer import ReplayBuffer


def mlp(input_dim, hidden_dims, output_dim, activation=nn.ReLU, out_activation=None):
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation())
        prev_dim = h
    layers.append(nn.Linear(prev_dim, output_dim))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden_dims=(256, 256)):
        super().__init__()
        self.net = mlp(
            obs_dim, hidden_dims, act_dim, activation=nn.ReLU, out_activation=nn.Tanh
        )
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    """Q(s,a): recebe observação e ação concatenadas."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q = mlp(
            obs_dim + act_dim, hidden_dims, 1, activation=nn.ReLU, out_activation=None
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


class TD3:
    """
    TD3 “puro”:
      - Actor determinístico
      - Dois críticos (Q1, Q2)
      - Target networks (actor_target, q1_target, q2_target)
      - Target policy smoothing
      - Policy delay
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_limit: float,
        actor_hidden_dims=(256, 256),
        critic_hidden_dims=(256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        policy_delay: int = 2,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        device: str = "cuda",
        target_noise: float = 0.2,
        target_noise_clip: float = 0.5,
    ):
        self.device = device

        self.actor = Actor(obs_dim, act_dim, act_limit, actor_hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.q1 = Critic(obs_dim, act_dim, critic_hidden_dims).to(device)
        self.q2 = Critic(obs_dim, act_dim, critic_hidden_dims).to(device)
        self.q1_target = copy.deepcopy(self.q1).to(device)
        self.q2_target = copy.deepcopy(self.q2).to(device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.act_limit = act_limit

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr
        )

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, obs, noise_scale: float = 0.0):
        """
        obs: np.ndarray shape (obs_dim,) ou (batch, obs_dim)
        """
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        act = self.actor(obs)
        if noise_scale > 0.0:
            act += noise_scale * torch.randn_like(act)
        act = torch.clamp(act, -self.act_limit, self.act_limit)
        return act.cpu().numpy()

    def soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def train_step(self, replay_buffer: ReplayBuffer, batch_size: int = 256):
        """
        Um passo de atualização do TD3.
        Retorna um dict de métricas (pode ser usado no logger).
        """
        self.total_it += 1

        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_act = self.actor_target(next_obs) + noise
            next_act = next_act.clamp(-self.act_limit, self.act_limit)

            q1_next = self.q1_target(next_obs, next_act)
            q2_next = self.q2_target(next_obs, next_act)
            q_next_min = torch.min(q1_next, q2_next)

            target_q = rew + self.gamma * (1.0 - done) * q_next_min

        q1_pred = self.q1(obs, act)
        q2_pred = self.q2(obs, act)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        q_loss = q1_loss + q2_loss

        self.q_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0
        )
        self.q_opt.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.policy_delay == 0:
            pi = self.actor(obs)
            actor_loss = -self.q1(obs, pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_opt.step()

            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.q1, self.q1_target)
            self.soft_update(self.q2, self.q2_target)

        info = {
            "q_loss": q_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }
        return info


@dataclass
class TD3Config:
    obs_dim: int
    act_dim: int
    hidden_dims: list[int]
    gamma: float
    tau: float
    actor_lr: float
    critic_lr: float
    policy_delay: int
    target_noise: float
    target_noise_clip: float
    exploration_noise: float
    device: str
    action_scale: float = 1.0


class TD3Agent:
    def __init__(self, cfg: TD3Config):
        self.config = cfg
        self.cfg = cfg

        self.device = torch.device(cfg.device)

        self.actor = Actor(
            cfg.obs_dim, cfg.act_dim, cfg.action_scale, cfg.hidden_dims
        ).to(self.device)

        self.actor_target = copy.deepcopy(self.actor).to(self.device)

        self.q1 = Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden_dims).to(self.device)
        self.q2 = Critic(cfg.obs_dim, cfg.act_dim, cfg.hidden_dims).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=cfg.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=cfg.critic_lr)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.policy_delay = cfg.policy_delay
        self.target_noise = cfg.target_noise
        self.target_noise_clip = cfg.target_noise_clip
        self.exploration_noise = cfg.exploration_noise
        self.action_scale = cfg.action_scale

        self.total_it = 0

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        """
        Compatível com OffPolicyRunner e evaluate_agent.
        Retorna tensor, não numpy.
        """
        obs = obs.to(self.device)
        action = self.actor(obs)

        if not eval_mode:
            noise = self.exploration_noise * torch.randn_like(action)
            action = action + noise
            action = torch.clamp(action, -self.action_scale, self.action_scale)

        return action

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Compatível com OffPolicyRunner.
        Recebe batch como dict e retorna dict de métricas.
        """
        self.total_it += 1

        obs = batch["obs"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        dones = batch["dones"].to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.target_noise).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_actions = self.actor_target(next_obs) + noise
            next_actions = next_actions.clamp(-self.action_scale, self.action_scale)

            q1_next = self.q1_target(next_obs, next_actions)
            q2_next = self.q2_target(next_obs, next_actions)
            q_next_min = torch.min(q1_next, q2_next)

            target_q = rewards + self.gamma * (1.0 - dones) * q_next_min

        q1_pred = self.q1(obs, actions)
        q2_pred = self.q2(obs, actions)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        q_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.q1.parameters()) + list(self.q2.parameters()), max_norm=1.0
        )
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        actor_loss = torch.tensor(0.0, device=self.device)
        if self.total_it % self.policy_delay == 0:
            pi = self.actor(obs)
            actor_loss = -self.q1(obs, pi).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

        return {
            "q_loss": q_loss.item(),
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "actor_loss": actor_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
        }

    def _soft_update(self, source, target):
        """Soft update target networks."""
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

    def select_action(self, obs, deterministic: bool = False):
        """Método legado - retorna numpy."""
        noise = 0.0 if deterministic else self.exploration_noise
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        action = self.act(
            torch.as_tensor(obs, device=self.device), eval_mode=deterministic
        )
        return action.cpu().numpy()

    def train_step(self, replay_buffer, batch_size: int):
        """Método legado - compatível com uso direto."""
        batch = replay_buffer.sample_batch(batch_size)

        batch_dict = {
            "obs": batch["obs"],
            "actions": batch["act"],
            "rewards": batch["rew"],
            "next_obs": batch["next_obs"],
            "dones": batch["done"],
        }
        return self.update(batch_dict)
