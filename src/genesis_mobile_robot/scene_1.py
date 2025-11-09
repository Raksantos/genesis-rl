from pathlib import Path
import numpy as np
import genesis as gs


def add_wall_bounds(scene, lower, upper):
    """Cria uma parede fixa usando bounds absolutos."""
    scene.add_entity(
        gs.morphs.Box(
            lower=lower,
            upper=upper,
            fixed=True,
        )
    )


def main():
    gs.init(backend=gs.cpu)

    scene = gs.Scene(
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(8.0, 0.0, 5.0),
            camera_lookat=(0.0, 0.0, 0.0),
        ),
    )

    # chão
    scene.add_entity(gs.morphs.Plane())

    # ===== arena =====
    half_x = 5.0
    half_y = 5.0
    wall_thickness = 0.15
    wall_height = 1.5

    # topo (y = +5)
    add_wall_bounds(
        scene,
        lower=(-half_x, half_y - wall_thickness, 0.0),
        upper=(half_x, half_y, wall_height),
    )
    # base (y = -5)
    add_wall_bounds(
        scene,
        lower=(-half_x, -half_y, 0.0),
        upper=(half_x, -half_y + wall_thickness, wall_height),
    )
    # direita (x = +5)
    add_wall_bounds(
        scene,
        lower=(half_x - wall_thickness, -half_y, 0.0),
        upper=(half_x, half_y, wall_height),
    )
    # esquerda (x = -5)
    add_wall_bounds(
        scene,
        lower=(-half_x, -half_y, 0.0),
        upper=(-half_x + wall_thickness, half_y, wall_height),
    )

    # obstáculo interno
    add_wall_bounds(
        scene,
        lower=(-1.0, -0.1, 0.0),
        upper=(1.0, 0.1, 1.0),
    )

    # ===== robô =====
    project_root = Path(__file__).resolve().parents[2]
    mjcf_path = project_root / "xml" / "mobile_base" / "diff_drive.xml"

    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=str(mjcf_path),
            pos=(-4.0, -4.0, 0.0),
        )
    )

    # ===== goal =====
    scene.add_entity(
        gs.morphs.Box(
            pos=(4.0, 4.0, 0.05),
            size=(0.15, 0.15, 0.08),
            fixed=True,
        )
    )

    scene.build()

    # atuadores
    try:
        left_id = robot.get_actuator_id("left_wheel_act")
        right_id = robot.get_actuator_id("right_wheel_act")
    except Exception:
        left_id = right_id = None

    for _ in range(3000):
        if left_id is not None and right_id is not None:
            ctrl = np.zeros(robot.num_actuators, dtype=np.float32)
            ctrl[left_id] = 4.0
            ctrl[right_id] = 4.0
            robot.set_control(ctrl)
        scene.step()


if __name__ == "__main__":
    main()
