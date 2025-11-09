from pathlib import Path
import genesis as gs


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(show_viewer=True)
    scene.add_entity(gs.morphs.Plane())

    project_root = Path(__file__).resolve().parents[2]
    mjcf_path = project_root / "genesis-rl" / "xml" / "mobile_base" / "scene_1.xml"

    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(mjcf_path),
            pos=(0, 0, 0),
        )
    )

    scene.build()

    try:
        left_id = robot.get_actuator_id("left_wheel_act")
        right_id = robot.get_actuator_id("right_wheel_act")
    except Exception:
        left_id = right_id = None

    for _ in range(2000):
        if left_id is not None and right_id is not None:
            import numpy as np

            ctrl = np.zeros(robot.num_actuators, dtype=np.float32)
            ctrl[left_id] = 5.0
            ctrl[right_id] = 5.0
            robot.set_control(ctrl)

        scene.step()


if __name__ == "__main__":
    main()
