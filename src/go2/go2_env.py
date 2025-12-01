import torch
import math
import numpy as np
import genesis as gs
from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


def generate_random_terrain_heightmap(
    size: tuple[float, float],
    resolution: tuple[int, int],
    height_range: tuple[float, float],
    num_functions: int = 8,
    seed: int = None,
) -> np.ndarray:
    """
    Gera um heightmap randômico usando múltiplas funções matemáticas.
    Cria elevações suaves como colinas, vales e ondulações.

    Args:
        size: Tamanho do terreno em metros (width, length)
        resolution: Resolução do heightmap (width_res, length_res)
        height_range: Range de altura (min, max)
        num_functions: Número de funções matemáticas para combinar
        seed: Seed para reprodutibilidade

    Returns:
        heightmap: Array numpy com as alturas do terreno
    """
    if seed is not None:
        np.random.seed(seed)

    width, length = size
    width_res, length_res = resolution
    min_height, max_height = height_range

    # Cria grade de coordenadas
    x = np.linspace(-width / 2, width / 2, width_res)
    y = np.linspace(-length / 2, length / 2, length_res)
    X, Y = np.meshgrid(x, y)

    # Inicializa o heightmap
    heightmap = np.zeros((length_res, width_res))

    # Combina múltiplas funções matemáticas para criar elevações variadas
    for i in range(num_functions):
        # Prioriza funções que criam elevações suaves
        # Reduz probabilidade de ruído puro
        func_weights = [
            2.0,
            2.0,
            2.0,
            3.0,
            2.5,
            2.0,
            1.5,
            0.5,
        ]  # Mais peso para elevações
        func_types = [
            "gaussian",  # Colinas suaves
            "sin",  # Ondulações
            "cos",  # Ondulações
            "sin_cos",  # Padrões complexos
            "ripple",  # Ondas circulares
            "wave",  # Ondas direcionais
            "mountain",  # Montanhas (nova)
            "noise",  # Ruído suave
        ]

        func_type = np.random.choice(
            func_types, p=np.array(func_weights) / np.sum(func_weights)
        )

        # Parâmetros aleatórios ajustados para criar elevações mais naturais
        freq_x = np.random.uniform(0.05, 1.5)  # Frequências mais baixas para suavidade
        freq_y = np.random.uniform(0.05, 1.5)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.4, 1.2)  # Amplitudes variadas
        offset_x = np.random.uniform(-width / 3, width / 3)
        offset_y = np.random.uniform(-length / 3, length / 3)

        if func_type == "sin":
            contribution = (
                amplitude
                * np.sin(freq_x * (X - offset_x) + phase_x)
                * np.sin(freq_y * (Y - offset_y) + phase_y)
            )
        elif func_type == "cos":
            contribution = (
                amplitude
                * np.cos(freq_x * (X - offset_x) + phase_x)
                * np.cos(freq_y * (Y - offset_y) + phase_y)
            )
        elif func_type == "sin_cos":
            contribution = amplitude * (
                np.sin(freq_x * (X - offset_x) + phase_x)
                * np.cos(freq_y * (Y - offset_y) + phase_y)
            )
        elif func_type == "gaussian":
            # Colinas suaves
            sigma_x = width / (3 * freq_x)
            sigma_y = length / (3 * freq_y)
            contribution = amplitude * np.exp(
                -(
                    (X - offset_x) ** 2 / (2 * sigma_x**2)
                    + (Y - offset_y) ** 2 / (2 * sigma_y**2)
                )
            )
        elif func_type == "ripple":
            # Ondas circulares suaves
            r = np.sqrt((X - offset_x) ** 2 + (Y - offset_y) ** 2)
            contribution = amplitude * np.sin(freq_x * r + phase_x) / (1 + r * 0.3)
        elif func_type == "wave":
            # Ondas direcionais
            contribution = amplitude * np.sin(
                freq_x * (X - offset_x) + freq_y * (Y - offset_y) + phase_x
            )
        elif func_type == "mountain":
            # Montanhas usando múltiplas gaussianas
            r = np.sqrt((X - offset_x) ** 2 + (Y - offset_y) ** 2)
            sigma = width / (2 * freq_x)
            contribution = amplitude * np.exp(-(r**2) / (2 * sigma**2))
            # Adiciona rugosidade
            contribution += (
                0.3
                * amplitude
                * np.sin(3 * freq_x * r + phase_x)
                * np.exp(-(r**2) / (4 * sigma**2))
            )
        else:  # noise suave
            # Ruído suavizado usando filtro gaussiano simples
            noise = np.random.randn(length_res, width_res)
            # Aplica suavização simples se scipy estiver disponível
            try:
                from scipy import ndimage

                noise = ndimage.gaussian_filter(noise, sigma=2.0)
            except ImportError:
                # Se scipy não estiver disponível, aplica média simples manual
                # Aplica média móvel simples
                kernel_size = 5
                pad = kernel_size // 2
                noise_padded = np.pad(noise, pad, mode="edge")
                noise_smooth = np.zeros_like(noise)
                for i in range(length_res):
                    for j in range(width_res):
                        noise_smooth[i, j] = np.mean(
                            noise_padded[i : i + kernel_size, j : j + kernel_size]
                        )
                noise = noise_smooth
            contribution = amplitude * noise * 0.15

        # Aplica peso decrescente para funções posteriores (cria detalhes finos)
        weight = 1.0 / (1.0 + i * 0.2)
        heightmap += contribution * weight

    # Aplica suavização final para reduzir pontas
    try:
        from scipy import ndimage

        heightmap = ndimage.gaussian_filter(heightmap, sigma=1.0)
    except ImportError:
        # Se scipy não estiver disponível, aplica média móvel simples
        kernel_size = 3
        pad = kernel_size // 2
        heightmap_padded = np.pad(heightmap, pad, mode="edge")
        heightmap_smooth = np.zeros_like(heightmap)
        for i in range(length_res):
            for j in range(width_res):
                heightmap_smooth[i, j] = np.mean(
                    heightmap_padded[i : i + kernel_size, j : j + kernel_size]
                )
        heightmap = heightmap_smooth

    # Normaliza para o range desejado
    heightmap = heightmap - heightmap.min()
    heightmap = heightmap / (heightmap.max() + 1e-8)
    heightmap = heightmap * (max_height - min_height) + min_height

    # Garante que o centro (0, 0) está na altura zero
    center_idx_x = width_res // 2
    center_idx_y = length_res // 2
    center_height = heightmap[center_idx_y, center_idx_x]
    heightmap = heightmap - center_height

    return heightmap


class Go2Env:
    def __init__(
        self,
        num_envs,
        env_cfg,
        obs_cfg,
        reward_cfg,
        command_cfg,
        show_viewer=False,
        add_camera=False,
    ):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=30,
            ),
            show_viewer=show_viewer,
        )

        # add terrain (random or plane)
        use_random_terrain = self.env_cfg.get("use_random_terrain", False)

        if use_random_terrain:
            # Generate random terrain using mathematical functions
            terrain_size = tuple(self.env_cfg.get("terrain_size", (20.0, 20.0)))
            terrain_resolution = tuple(
                self.env_cfg.get("terrain_resolution", (200, 200))
            )
            terrain_height_range = tuple(
                self.env_cfg.get("terrain_height_range", (-0.3, 0.3))
            )
            terrain_num_functions = self.env_cfg.get("terrain_num_functions", 5)

            # Generate heightmap (in meters)
            heightmap_meters = generate_random_terrain_heightmap(
                size=terrain_size,
                resolution=terrain_resolution,
                height_range=terrain_height_range,
                num_functions=terrain_num_functions,
                seed=None,  # Random seed each time
            )

            # Calculate scales for Terrain
            # horizontal_scale: size of each cell in meters (distance between grid points)
            # vertical_scale: meters per height unit
            width, length = terrain_size
            width_res, length_res = terrain_resolution
            # If we have n points, we have n-1 intervals covering the width
            horizontal_scale = width / (width_res - 1) if width_res > 1 else width
            vertical_scale = 1.0  # heightmap is already in meters

            # Position terrain so that center is at (0, 0, 0)
            # The terrain's origin is at its corner, so we offset by half the size
            terrain_pos = (-width / 2, -length / 2, 0.0)

            # Create terrain using Terrain with height_field
            # When height_field is specified, other terrain configs are ignored
            # uv_scale controls texture tiling: higher values = smaller texture, more repetition
            terrain_uv_scale = self.env_cfg.get(
                "terrain_uv_scale", 10.0
            )  # Default: 5.0 for more repetition
            terrain_morph = gs.morphs.Terrain(
                height_field=heightmap_meters,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                pos=terrain_pos,  # Position terrain so center is at (0, 0, 0)
                uv_scale=terrain_uv_scale,  # Higher values = smaller texture, more repetition
            )

            # Load checker texture and apply via surface parameter
            # According to Genesis docs: use diffuse_texture parameter with ImageTexture
            try:
                checker_texture = gs.textures.ImageTexture(
                    image_path="textures/checker.png"
                )
                surface = gs.surfaces.Rough(diffuse_texture=checker_texture)
                terrain_entity = self.scene.add_entity(
                    morph=terrain_morph,
                    surface=surface,
                )
                print("Applied checker texture to terrain via diffuse_texture")
            except Exception as e:
                # Fallback: create terrain without texture
                print(f"Could not apply texture: {e}")
                terrain_entity = self.scene.add_entity(morph=terrain_morph)

            self.terrain = terrain_entity

            # Calculate height at center (0, 0) to position robot correctly
            # Since we normalized the heightmap to have center at 0, center_height should be 0
            center_idx_x = terrain_resolution[0] // 2
            center_idx_y = terrain_resolution[1] // 2
            center_height = float(heightmap_meters[center_idx_y, center_idx_x])

            # Get base height from config (default 0.42 for Go2)
            base_height = self.env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])[2]

            # Robot initial position: center of terrain at terrain height + base height
            robot_init_pos = [0.0, 0.0, center_height + base_height]
        else:
            # Use simple plane
            self.scene.add_entity(
                gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True)
            )
            robot_init_pos = self.env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])
            self.terrain = None

        # add robot
        self.base_init_pos = torch.tensor(robot_init_pos, device=gs.device)
        self.base_init_quat = torch.tensor(
            self.env_cfg["base_init_quat"], device=gs.device
        )
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        if add_camera:
            self.cam_0 = self.scene.add_camera(
                res=(1920, 1080),
                pos=(2.5, 0.5, 3.5),
                lookat=(0, 0, 0.5),
                fov=40,
                GUI=True,
            )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [
            self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]
        ]

        # PD control parameters
        self.robot.set_dofs_kp(
            [self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx
        )
        self.robot.set_dofs_kv(
            [self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx
        )

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )

        # initialize buffers
        self.base_lin_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_ang_vel = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.projected_gravity = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.global_gravity = torch.tensor(
            [0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float
        )
        self.rew_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_int
        )
        self.commands = torch.zeros(
            (self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float
        )
        self.commands_scale = torch.tensor(
            [
                self.obs_scales["lin_vel"],
                self.obs_scales["lin_vel"],
                self.obs_scales["ang_vel"],
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float
        )
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros(
            (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
        )
        self.base_quat = torch.zeros(
            (self.num_envs, 4), device=gs.device, dtype=gs.tc_float
        )
        self.default_dof_pos = torch.tensor(
            [
                self.env_cfg["default_joint_angles"][name]
                for name in self.env_cfg["joint_names"]
            ],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        if len(envs_idx) == 0:
            return

        self.commands[envs_idx, 0] = gs_rand_float(
            *self.command_cfg["lin_vel_x_range"],
            (len(envs_idx),),
            gs.device,
        ).to(self.commands.dtype)

        self.commands[envs_idx, 1] = gs_rand_float(
            *self.command_cfg["lin_vel_y_range"],
            (len(envs_idx),),
            gs.device,
        ).to(self.commands.dtype)

        self.commands[envs_idx, 2] = gs_rand_float(
            *self.command_cfg["ang_vel_range"],
            (len(envs_idx),),
            gs.device,
        ).to(self.commands.dtype)

    def step(self, actions):
        self.actions = torch.clip(
            actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"]
        )
        exec_actions = (
            self.last_actions if self.simulate_action_latency else self.actions
        )
        target_dof_pos = (
            exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        )
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(
                torch.ones_like(self.base_quat) * self.inv_base_init_quat,
                self.base_quat,
            ),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)

        # resample commands
        envs_idx = (
            (
                self.episode_length_buf
                % int(self.env_cfg["resampling_time_s"] / self.dt)
                == 0
            )
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 1])
            > self.env_cfg["termination_if_pitch_greater_than"]
        )
        self.reset_buf |= (
            torch.abs(self.base_euler[:, 0])
            > self.env_cfg["termination_if_roll_greater_than"]
        )

        time_out_idx = (
            (self.episode_length_buf > self.max_episode_length)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self.extras["time_outs"] = torch.zeros_like(
            self.reset_buf, device=gs.device, dtype=gs.tc_float
        )
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        self.obs_buf = torch.cat(
            [
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                (self.dof_pos - self.default_dof_pos)
                * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
            ],
            axis=-1,
        )

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(
            self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.robot.set_quat(
            self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx
        )
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item()
                / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(
            torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1
        )
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])
