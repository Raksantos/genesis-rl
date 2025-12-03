from dataclasses import asdict, dataclass, field


@dataclass
class EnvCfg:
    num_actions: int = 12
    default_joint_angles: dict[str, float] = field(
        default_factory=lambda: {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        }
    )
    joint_names: list[str] = field(
        default_factory=lambda: [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]
    )
    kp: float = 20.0
    kd: float = 0.5
    termination_if_roll_greater_than: float = 10.0
    termination_if_pitch_greater_than: float = 10.0
    base_init_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.42])
    base_init_quat: list[float] = field(default_factory=lambda: [1.0, 0.0, 0.0, 0.0])
    episode_length_s: float = 20.0
    resampling_time_s: float = 4.0
    action_scale: float = 0.25
    simulate_action_latency: bool = True
    clip_actions: float = 100.0
    # Terrain configuration
    use_random_terrain: bool = True  # Use random terrain by default
    terrain_size: tuple[float, float] = (
        150.0,
        150.0,
    )  # (width, length) in meters - same size as plane.urdf
    terrain_resolution: tuple[int, int] = (
        100,
        100,
    )  # (width_res, length_res) - balanced for performance and quality
    terrain_height_range: tuple[float, float] = (
        -1.0,
        1.0,
    )  # min and max height variation
    terrain_num_functions: int = 8  # number of mathematical functions to combine
    terrain_uv_scale: float = (
        50.0  # UV scale for texture tiling - higher values = smaller texture, more repetition
    )
    # Goal/waypoint configuration
    use_goal_navigation: bool = True  # Enable goal-based navigation rewards
    goal_reach_distance: float = (
        0.5  # Distance threshold to consider goal reached (meters)
    )
    goal_resample_time_s: float = 10.0  # Time interval to resample goals (seconds)


@dataclass
class ObsCfg:
    num_obs: int = 45
    obs_scales: dict[str, float] = field(
        default_factory=lambda: {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        }
    )


@dataclass
class RewardCfg:
    tracking_sigma: float = 0.25
    base_height_target: float = 0.3
    feet_height_target: float = 0.075
    reward_scales: dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
            "goal_distance": 2.0,  # Reward for reducing distance to goal
            "goal_velocity": 1.0,  # Reward for velocity towards goal
            "goal_reached": 10.0,  # Reward for reaching goal
        }
    )


@dataclass
class CommandCfg:
    num_commands: int = 3
    lin_vel_x_range: list[float] = field(default_factory=lambda: [0.5, 0.5])
    lin_vel_y_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
    ang_vel_range: list[float] = field(default_factory=lambda: [0.0, 0.0])


def get_cfgs():
    env_cfg = asdict(EnvCfg())
    obs_cfg = asdict(ObsCfg())
    reward_cfg = asdict(RewardCfg())
    command_cfg = asdict(CommandCfg())
    return env_cfg, obs_cfg, reward_cfg, command_cfg
