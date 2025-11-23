import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch

from src.go2 import Go2Env


class Go2GymEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(
        self,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        device="cuda:0",
        show_viewer=False,
    ):
        super().__init__()

        self.device = torch.device(device)

        self.go2_env = Go2Env(
            num_envs=1,
            env_cfg=env_cfg,
            obs_cfg=obs_cfg,
            reward_cfg=reward_cfg,
            command_cfg=command_cfg,
            show_viewer=show_viewer,
        )

        self.num_obs = self.go2_env.num_obs
        self.num_actions = self.go2_env.num_actions

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_obs,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        obs, _ = self.go2_env.reset()

        obs = obs[0].cpu().numpy().astype(np.float32)

        info = {}
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action_t = torch.tensor(action, device=self.device).unsqueeze(0)

        with torch.no_grad():
            next_obs, rewards, dones, infos = self.go2_env.step(action_t)

        obs_np = next_obs[0].cpu().numpy().astype(np.float32)
        reward_np = float(rewards[0].cpu().item())
        done_np = bool(dones[0].cpu().item())

        terminated = done_np
        truncated = False

        info = {}

        return obs_np, reward_np, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
