import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
            obs_dim,
            hidden_dims,
            act_dim,
            activation=nn.ReLU,
            out_activation=nn.Tanh,
        )
        self.act_limit = act_limit

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    """Q(s,a): recebe observação e ação concatenadas."""

    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.q = mlp(
            obs_dim + act_dim,
            hidden_dims,
            1,
            activation=nn.ReLU,
            out_activation=None,
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return self.q(x).squeeze(-1)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device="cpu"):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def add(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=self.device),
            act=torch.as_tensor(self.acts_buf[idxs], device=self.device),
            rew=torch.as_tensor(self.rews_buf[idxs], device=self.device),
            next_obs=torch.as_tensor(self.next_obs_buf[idxs], device=self.device),
            done=torch.as_tensor(self.done_buf[idxs], device=self.device),
        )
        return batch


class DDPG:
    """
    DDPG “puro”:
      - Actor determinístico
      - Um crítico Q(s,a)
      - Redes alvo para actor e critic
      - Exploração via ruído aditivo (Gaussian ou OU, aqui Gaussian simples)
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
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        max_grad_norm: float = 1.0,
        device: str = "cuda",
    ):
        self.device = device

        self.actor = Actor(obs_dim, act_dim, act_limit, actor_hidden_dims).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        self.critic = Critic(obs_dim, act_dim, critic_hidden_dims).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.gamma = gamma
        self.tau = tau
        self.act_limit = act_limit
        self.max_grad_norm = max_grad_norm

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.total_it = 0

    @torch.no_grad()
    def select_action(self, obs, noise_scale: float = 0.0):
        """
        obs: np.ndarray shape (obs_dim,) ou (batch, obs_dim).
        Retorna np.ndarray com mesma shape de batch na dimensão de ações.
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
        Um passo de atualização do DDPG.
        Retorna um dict com métricas para logging.
        """
        self.total_it += 1

        batch = replay_buffer.sample_batch(batch_size)
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_act = self.actor_target(next_obs)
            target_q = self.critic_target(next_obs, next_act)
            target_q = rew + self.gamma * (1.0 - done) * target_q

        q_pred = self.critic(obs, act)
        critic_loss = F.mse_loss(q_pred, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        actor_loss = -self.critic(obs, self.actor(obs)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        self.soft_update(self.actor, self.actor_target)
        self.soft_update(self.critic, self.critic_target)

        info = {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_mean": q_pred.mean().item(),
        }
        return info
