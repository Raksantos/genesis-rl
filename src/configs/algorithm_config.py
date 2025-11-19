from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AlgorithmCfg:
    class_name: str = "PPO"
    clip_param: float = 0.2
    desired_kl: float = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 0.001
    max_grad_norm: float = 1.0
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    use_clipped_value_loss: bool = True
    value_loss_coef: float = 1.0


@dataclass
class PolicyCfg:
    class_name: str = "ActorCritic"
    activation: str = "elu"
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    init_noise_std: float = 1.0


@dataclass
class RunnerInnerCfg:
    algorithm_class_name: str = "PPO"
    checkpoint: int = -1
    experiment_name: str = ""
    load_run: int = -1
    log_interval: int = 1
    max_iterations: int = 10_000
    num_steps_per_env: int = 24
    policy_class_name: str = "ActorCritic"
    record_interval: int = -1
    resume: bool = False
    resume_path: str | None = None
    run_name: str = ""
    runner_class_name: str = "runner_class_name"
    save_interval: int = 100


@dataclass
class TrainCfg:
    algorithm: AlgorithmCfg
    policy: PolicyCfg
    runner: RunnerInnerCfg
    init_member_classes: dict[str, Any] = field(default_factory=dict)
    runner_class_name: str = "OnPolicyRunner"
    num_steps_per_env: int = 24
    save_interval: int = 100
    obs_groups: Any = None
    seed: int = 42


def get_train_cfg(exp_name: str, max_iterations: int) -> dict[str, Any]:
    algorithm = AlgorithmCfg()
    policy = PolicyCfg()
    runner = RunnerInnerCfg(experiment_name=exp_name, max_iterations=max_iterations)

    train_cfg = TrainCfg(
        algorithm=algorithm,
        policy=policy,
        runner=runner,
    )

    return asdict(train_cfg)
