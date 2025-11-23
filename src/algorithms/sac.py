from dataclasses import dataclass, field
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation=nn.ReLU,
    ):
        super().__init__()
        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim

        layers.append(nn.Linear(last_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        log_std_bounds=(-20, 2),
    ):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dims, act_dim * 2)
        self.log_std_bounds = log_std_bounds
        self.act_dim = act_dim

    def forward(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean_log_std = self.net(obs)
        mean, log_std = mean_log_std.chunk(2, dim=-1)

        log_std = torch.clamp(log_std, *self.log_std_bounds)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        log_prob = normal.log_prob(z).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)

        return action, log_prob, mean


@dataclass
class SACConfig:
    obs_dim: int
    act_dim: int
    hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float | None = None
    device: str = "cuda"
    action_scale: float = 1.0


class SACAgent:
    def __init__(self, config: SACConfig):
        self.config = config

        self.device = torch.device(config.device)

        self.actor = GaussianPolicy(
            obs_dim=config.obs_dim,
            act_dim=config.act_dim,
            hidden_dims=list(config.hidden_dims),
        ).to(self.device)

        critic_input_dim = config.obs_dim + config.act_dim
        self.q1 = MLP(critic_input_dim, list(config.hidden_dims), 1).to(self.device)
        self.q2 = MLP(critic_input_dim, list(config.hidden_dims), 1).to(self.device)

        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        for param in self.q1_target.parameters():
            param.requires_grad = False

        for param in self.q2_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.q1_optimizer = torch.optim.Adam(self.q1.parameters(), lr=config.critic_lr)
        self.q2_optimizer = torch.optim.Adam(self.q2.parameters(), lr=config.critic_lr)

        if config.target_entropy is None:
            config.target_entropy = -float(config.act_dim)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, eval_mode: bool = False) -> torch.Tensor:
        obs = obs.to(self.device)
        if eval_mode:
            mean_logstd = self.actor.net(obs)
            mean, _ = mean_logstd.chunk(2, dim=-1)
            action = torch.tanh(mean)

        else:
            action, _, _ = self.actor(obs)

        return action * self.config.action_scale

    def _soft_update(self, net: nn.Module, target_net: nn.Module):
        tau = self.config.tau
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * param.data)

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor(next_obs)
            next_action = next_action * self.config.action_scale

            q1_next = self.q1_target(torch.cat([next_obs, next_action], dim=-1))
            q2_next = self.q2_target(torch.cat([next_obs, next_action], dim=-1))
            q_target_min = torch.min(q1_next, q2_next)

            target_v = q_target_min - self.alpha * next_log_prob
            target_q = rewards + (1 - dones) * self.config.gamma * target_v

        q1_pred = self.q1(torch.cat([obs, actions], dim=-1))
        q2_pred = self.q2(torch.cat([obs, actions], dim=-1))

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)
        critic_loss = q1_loss + q2_loss

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        critic_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        new_action, log_prob, _ = self.actor(obs)
        new_action = new_action * self.config.action_scale
        q1_pi = self.q1(torch.cat([obs, new_action], dim=-1))
        q2_pi = self.q2(torch.cat([obs, new_action], dim=-1))
        q_pi_min = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_prob - q_pi_min).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = -(
            self.log_alpha * (log_prob + self.config.target_entropy).detach()
        ).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        self._soft_update(self.q1, self.q1_target)
        self._soft_update(self.q2, self.q2_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }
