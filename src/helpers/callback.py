from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class EarlyStopCallback(BaseCallback):
    def __init__(self, patience: int = 10, min_delta: float = 0.0, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -np.inf
        self.num_bad_epochs = 0

    def _on_step(self) -> bool:
        try:
            ep_rew_mean = self.logger.name_to_value.get("rollout/ep_rew_mean", None)
        except Exception:
            ep_rew_mean = None

        if ep_rew_mean is None:
            return True

        if ep_rew_mean > self.best_mean_reward + self.min_delta:
            self.best_mean_reward = ep_rew_mean
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.verbose > 0:
            print(
                f"[EarlyStop] mean_reward={ep_rew_mean:.2f}, "
                f"best={self.best_mean_reward:.2f}, "
                f"bad_epochs={self.num_bad_epochs}/{self.patience}"
            )

        if self.num_bad_epochs >= self.patience:
            if self.verbose > 0:
                print("[EarlyStop] Parando treino por estagnação de recompensa.")
            return False

        return True
