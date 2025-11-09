from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.genesis_mobile_robot.env_diff_drive import GenesisDiffDriveGoalEnv


def make_env(show_viewer=False):
    project_root = Path(__file__).resolve().parents[2]
    xml_path = project_root / "xml" / "mobile_base" / "diff_drive.xml"

    env = GenesisDiffDriveGoalEnv(
        xml_path=str(xml_path),
        arena_half=5.0,
        show_viewer=show_viewer,
        max_steps=300,
    )
    return env


def main():
    env = make_env(show_viewer=True)

    check_env(env, warn=True)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="./ppo_genesis_tensorboard/",
    )

    model.learn(total_timesteps=100_000)

    model.save("ppo_genesis_diff_drive_model")


if __name__ == "__main__":
    main()
