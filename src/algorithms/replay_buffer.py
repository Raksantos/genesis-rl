import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, size: int, device: str = "cuda:0"):
        self.device = torch.device(device)
        self.max_size = size
        self.ptr = 0
        self.size = 0

        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)

    def store_batch(self, obs, acts, rews, next_obs, dones):
        """
        obs, acts, rews, next_obs, dones: np.array com shape (N, ...)
        """
        N = obs.shape[0]
        assert (
            acts.shape[0] == N
            and rews.shape[0] == N
            and next_obs.shape[0] == N
            and dones.shape[0] == N
        )

        idxs = np.arange(self.ptr, self.ptr + N) % self.max_size

        self.obs_buf[idxs] = obs
        self.acts_buf[idxs] = acts
        self.rews_buf[idxs] = rews
        self.next_obs_buf[idxs] = next_obs
        self.done_buf[idxs] = dones

        self.ptr = (self.ptr + N) % self.max_size
        self.size = min(self.size + N, self.max_size)

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor]:
        assert self.size >= batch_size, (
            "Not enough samples in buffer to sample the batch."
        )

        idxs = np.random.randint(0, self.size, size=batch_size)

        batch = dict(
            obs=torch.tensor(
                self.obs_buf[idxs], dtype=torch.float32, device=self.device
            ),
            actions=torch.tensor(
                self.acts_buf[idxs], dtype=torch.float32, device=self.device
            ),
            rewards=torch.tensor(
                self.rews_buf[idxs], dtype=torch.float32, device=self.device
            ),
            next_obs=torch.tensor(
                self.next_obs_buf[idxs], dtype=torch.float32, device=self.device
            ),
            dones=torch.tensor(
                self.done_buf[idxs], dtype=torch.float32, device=self.device
            ),
        )

        return batch
