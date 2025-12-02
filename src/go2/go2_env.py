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
    # Prioriza elevações suaves sobre ruído
    for i in range(num_functions):
        # Prioriza funções que criam elevações suaves, reduz ruído
        func_weights = [
            3.0,  # gaussian - colinas suaves
            2.5,  # sin - ondulações
            2.5,  # cos - ondulações
            2.5,  # sin_cos - padrões complexos
            2.0,  # ripple - ondas circulares
            2.0,  # wave - ondas direcionais
            2.5,  # mountain - montanhas
            0.3,  # noise - muito menos peso para reduzir ruído
        ]
        func_types = [
            "gaussian",  # Colinas suaves
            "sin",  # Ondulações
            "cos",  # Ondulações
            "sin_cos",  # Padrões complexos
            "ripple",  # Ondas circulares
            "wave",  # Ondas direcionais
            "mountain",  # Montanhas
            "noise",  # Ruído aleatório
        ]

        func_type = np.random.choice(
            func_types, p=np.array(func_weights) / np.sum(func_weights)
        )

        # Parâmetros muito mais aleatórios para criar terreno mais variado
        freq_x = np.random.uniform(0.01, 3.0)  # Maior range de frequências
        freq_y = np.random.uniform(0.01, 3.0)
        phase_x = np.random.uniform(0, 2 * np.pi)
        phase_y = np.random.uniform(0, 2 * np.pi)
        amplitude = np.random.uniform(0.2, 2.0)  # Maior range de amplitudes

        # Coloca os centros das funções FORA do mapa para evitar buracos
        # Os offsets ficam em uma faixa fora dos limites do mapa
        margin = width * 0.2  # Margem de 20% do tamanho do mapa
        # Gera offset fora do mapa: antes de -width/2 ou depois de +width/2
        if np.random.rand() < 0.5:
            offset_x = np.random.uniform(-width * 1.5, -width / 2 - margin)
        else:
            offset_x = np.random.uniform(width / 2 + margin, width * 1.5)

        if np.random.rand() < 0.5:
            offset_y = np.random.uniform(-length * 1.5, -length / 2 - margin)
        else:
            offset_y = np.random.uniform(length / 2 + margin, length * 1.5)

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
        else:  # noise - suavizado e reduzido
            # Ruído muito suavizado para criar apenas textura sutil
            noise = np.random.randn(length_res, width_res)
            # Suavização forte para reduzir ruído
            sigma = np.random.uniform(2.0, 4.0)  # Mais suavização
            try:
                from scipy import ndimage

                noise = ndimage.gaussian_filter(noise, sigma=sigma)
            except ImportError:
                # Se scipy não estiver disponível, aplica média simples manual
                kernel_size = max(5, int(sigma * 2))
                pad = kernel_size // 2
                noise_padded = np.pad(noise, pad, mode="edge")
                noise_smooth = np.zeros_like(noise)
                for i in range(length_res):
                    for j in range(width_res):
                        noise_smooth[i, j] = np.mean(
                            noise_padded[i : i + kernel_size, j : j + kernel_size]
                        )
                noise = noise_smooth
            # Reduz muito a contribuição do ruído - apenas textura sutil
            contribution = amplitude * noise * np.random.uniform(0.05, 0.15)

        # Peso mais variado para criar mais diversidade
        # Às vezes funções posteriores têm mais peso (cria variação)
        weight = np.random.uniform(0.5, 1.5) / (1.0 + i * 0.15)
        heightmap += contribution * weight

    # Aplica suavização final para reduzir ruído e criar terreno mais suave
    smoothing_sigma = np.random.uniform(1.0, 2.0)  # Mais suavização para reduzir ruído
    try:
        from scipy import ndimage

        heightmap = ndimage.gaussian_filter(heightmap, sigma=smoothing_sigma)
    except ImportError:
        # Se scipy não estiver disponível, aplica média móvel simples
        kernel_size = max(3, int(smoothing_sigma * 2))
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

    # Cria uma região plana no centro onde o robô começa
    # Define o raio da região plana (em metros)
    flat_radius = 5.0  # 5 metros de raio plano no centro
    center_x = width_res // 2
    center_y = length_res // 2

    # Cria máscara gaussiana para suavizar a transição da região plana
    for i in range(length_res):
        for j in range(width_res):
            # Distância do centro em coordenadas do mundo
            world_x = x[j]  # x[j] já está em coordenadas do mundo
            world_y = y[i]  # y[i] já está em coordenadas do mundo
            dist_from_center = np.sqrt(world_x**2 + world_y**2)

            # Se estiver dentro do raio plano, achata o terreno
            if dist_from_center < flat_radius:
                # Fator de suavização: 1.0 no centro, 0.0 na borda do raio
                # Usa função suave (cos) para transição
                if dist_from_center < flat_radius * 0.7:
                    # Região completamente plana
                    blend_factor = 0.0
                else:
                    # Zona de transição suave
                    transition = (dist_from_center - flat_radius * 0.7) / (
                        flat_radius * 0.3
                    )
                    blend_factor = (1.0 - np.cos(transition * np.pi)) / 2.0

                # Interpola entre altura atual e zero (plano)
                heightmap[i, j] = heightmap[i, j] * blend_factor

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

        # Shared terrain configuration variables
        use_random_terrain = self.env_cfg.get("use_random_terrain", False)
        terrain_size = tuple(
            self.env_cfg.get("terrain_size", (100.0, 100.0))
        )  # (width, length) in meters
        terrain_uv_scale = self.env_cfg.get(
            "terrain_uv_scale", 50.0
        )  # UV scale for texture tiling - higher values = smaller texture, more repetition

        # Load checker texture (shared between random terrain and plane)
        checker_texture = None
        try:
            checker_texture = gs.textures.ImageTexture(
                image_path="textures/checker.png"
            )
        except Exception as e:
            print(f"Could not load checker texture: {e}")

        # add terrain (random or plane)
        if use_random_terrain:
            # Generate random terrain using mathematical functions
            terrain_resolution = tuple(
                self.env_cfg.get("terrain_resolution", (100, 100))
            )
            terrain_height_range = tuple(
                self.env_cfg.get("terrain_height_range", (-1.0, 1.0))
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
            terrain_morph = gs.morphs.Terrain(
                height_field=heightmap_meters,
                horizontal_scale=horizontal_scale,
                vertical_scale=vertical_scale,
                pos=terrain_pos,  # Position terrain so center is at (0, 0, 0)
                uv_scale=terrain_uv_scale,  # Higher values = smaller texture, more repetition
            )

            # Apply checker texture via surface parameter
            if checker_texture is not None:
                surface = gs.surfaces.Rough(diffuse_texture=checker_texture)
                terrain_entity = self.scene.add_entity(
                    morph=terrain_morph,
                    surface=surface,
                )
                print("Applied checker texture to random terrain via diffuse_texture")
            else:
                terrain_entity = self.scene.add_entity(morph=terrain_morph)

            self.terrain = terrain_entity

            # Store terrain data for height queries
            self.terrain_heightmap = heightmap_meters
            self.terrain_size = terrain_size
            self.terrain_resolution = terrain_resolution
            self.terrain_pos = terrain_pos
            self.terrain_horizontal_scale = horizontal_scale

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
            # Use plane primitive with checker texture
            width, length = terrain_size
            # Create plane primitive centered at (0, 0, 0)
            # Plane accepts plane_size parameter: (width, length)
            # tile_size controls texture repetition - convert uv_scale to tile_size
            # Higher uv_scale = more repetition = smaller tile_size
            tile_size_val = max(1, int(terrain_size[0] / terrain_uv_scale))
            plane_morph = gs.morphs.Plane(
                pos=(0.0, 0.0, 0.0),  # Center at origin
                plane_size=(width, length),  # Size of the plane
                tile_size=(
                    tile_size_val,
                    tile_size_val,
                ),  # Tile size for texture repetition
                fixed=True,  # Plane should be fixed (defined in morph)
            )

            # Apply checker texture via surface parameter
            if checker_texture is not None:
                surface = gs.surfaces.Rough(diffuse_texture=checker_texture)
                plane_entity = self.scene.add_entity(
                    morph=plane_morph,
                    surface=surface,
                )
                print("Applied checker texture to plane primitive via diffuse_texture")
            else:
                plane_entity = self.scene.add_entity(
                    morph=plane_morph,
                )

            self.terrain = plane_entity
            robot_init_pos = self.env_cfg.get("base_init_pos", [0.0, 0.0, 0.42])

            # Store terrain data (for compatibility with height queries)
            self.terrain_heightmap = None  # No heightmap for flat plane
            self.terrain_size = terrain_size  # Store shared size
            self.terrain_resolution = None  # No resolution for plane
            self.terrain_pos = (-width / 2, -length / 2, 0.0)  # For consistency
            self.terrain_horizontal_scale = None  # No scale for plane

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

        # Check if goal navigation is enabled (before build, so we can add marker)
        use_goal_navigation = self.env_cfg.get("use_goal_navigation", False)

        # Create visual marker for goal (only for first environment if multiple)
        # Must be created BEFORE scene.build()
        self.goal_marker = None
        if use_goal_navigation and num_envs > 0:
            # Create a sphere marker for the goal (only for visualization)
            sphere_morph = gs.morphs.Sphere(
                radius=0.1,  # 10cm radius sphere
                pos=(
                    0.0,
                    0.0,
                    1.0,
                ),  # Start at visible height, will be updated after build
                collision=False,  # No collision, just visual
            )
            self.goal_marker = self.scene.add_entity(
                morph=sphere_morph,
                surface=gs.surfaces.Emission(color=(1.0, 0.0, 0.0)),  # Bright red color
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

        # Buffer for terrain height at robot positions (for rewards)
        self.terrain_height_at_robot = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )

        # Goal/waypoint system
        self.use_goal_navigation = (
            use_goal_navigation  # Use the value we already checked
        )
        if self.use_goal_navigation:
            # Goal positions (x, y, z) - z will be set based on terrain height
            self.goal_positions = torch.zeros(
                (self.num_envs, 3), device=gs.device, dtype=gs.tc_float
            )
            # Distance to goal
            self.goal_distances = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )
            # Previous distance to goal (for reward calculation)
            self.prev_goal_distances = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )
            # Goal reached flag
            self.goal_reached = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=torch.bool
            )
            # Time since last goal resample
            self.goal_resample_timer = torch.zeros(
                (self.num_envs,), device=gs.device, dtype=gs.tc_float
            )
            self.goal_reach_distance = self.env_cfg.get("goal_reach_distance", 0.5)
            self.goal_resample_time_s = self.env_cfg.get("goal_resample_time_s", 10.0)
            self.goal_resample_interval = math.ceil(self.goal_resample_time_s / self.dt)
        else:
            # Initialize goal-related buffers as None/zeros if disabled
            self.goal_positions = None
            self.goal_distances = None
            self.prev_goal_distances = None
            self.goal_reached = None
            self.goal_resample_timer = None

    def _get_terrain_height_at_positions(self, positions):
        if self.terrain_heightmap is None:
            # Flat terrain - return zeros
            return torch.zeros(
                (positions.shape[0],), device=positions.device, dtype=gs.tc_float
            )

        # Convert positions to terrain-local coordinates
        # Terrain is positioned at terrain_pos, so we need to offset
        terrain_x = (
            positions[:, 0] - self.terrain_pos[0]
        )  # x relative to terrain origin
        terrain_y = (
            positions[:, 1] - self.terrain_pos[1]
        )  # y relative to terrain origin

        # Convert to heightmap indices
        width, length = self.terrain_size
        width_res, length_res = self.terrain_resolution

        # Clamp to terrain bounds
        terrain_x = torch.clamp(terrain_x, 0.0, width)
        terrain_y = torch.clamp(terrain_y, 0.0, length)

        # Convert to indices (interpolate between grid points)
        # horizontal_scale is the distance between grid points
        # For width_res points covering width, we have (width_res - 1) intervals
        idx_x = terrain_x / self.terrain_horizontal_scale
        idx_y = terrain_y / self.terrain_horizontal_scale

        # Clamp indices to valid range [0, width_res - 1] and [0, length_res - 1]
        idx_x = torch.clamp(idx_x, 0.0, float(width_res - 1))
        idx_y = torch.clamp(idx_y, 0.0, float(length_res - 1))

        # Bilinear interpolation: get floor and ceil indices
        idx_x_floor = idx_x.floor().long()
        idx_y_floor = idx_y.floor().long()
        idx_x_ceil = torch.min(
            idx_x_floor + 1,
            torch.tensor(width_res - 1, device=idx_x.device, dtype=torch.long),
        )
        idx_y_ceil = torch.min(
            idx_y_floor + 1,
            torch.tensor(length_res - 1, device=idx_y.device, dtype=torch.long),
        )

        # Get interpolation weights (fractional part)
        wx = (idx_x - idx_x_floor.float()).clamp(0.0, 1.0)
        wy = (idx_y - idx_y_floor.float()).clamp(0.0, 1.0)

        # Handle edge case: when exactly at the last point, weight should be 0
        # This ensures we don't interpolate beyond the heightmap bounds
        wx = torch.where(idx_x_floor >= width_res - 1, torch.zeros_like(wx), wx)
        wy = torch.where(idx_y_floor >= length_res - 1, torch.zeros_like(wy), wy)

        # Convert heightmap to torch tensor if needed
        if not isinstance(self.terrain_heightmap, torch.Tensor):
            heightmap_t = torch.from_numpy(self.terrain_heightmap).to(
                device=positions.device, dtype=gs.tc_float
            )
        else:
            heightmap_t = self.terrain_heightmap.to(device=positions.device)

        # Sample heights at corners (note: heightmap is [length_res, width_res])
        h00 = heightmap_t[idx_y_floor, idx_x_floor]
        h10 = heightmap_t[idx_y_floor, idx_x_ceil]
        h01 = heightmap_t[idx_y_ceil, idx_x_floor]
        h11 = heightmap_t[idx_y_ceil, idx_x_ceil]

        # Bilinear interpolation
        h0 = h00 * (1 - wx) + h10 * wx
        h1 = h01 * (1 - wx) + h11 * wx
        terrain_heights = h0 * (1 - wy) + h1 * wy

        return terrain_heights

    def _resample_goals(self, envs_idx):
        """Generate random goal positions within terrain bounds."""
        if len(envs_idx) == 0 or not self.use_goal_navigation:
            return

        # Get terrain bounds (or use default if flat terrain)
        if self.terrain_size is not None:
            width, length = self.terrain_size
            # Terrain is centered at (0, 0), so it goes from -width/2 to +width/2
            # Use a small margin (0.5m) to keep goals away from edges
            margin = 0.5
            min_x = -width / 2 + margin
            max_x = width / 2 - margin
            min_y = -length / 2 + margin
            max_y = length / 2 - margin
        else:
            # Default bounds for flat terrain
            min_x = -8.0
            max_x = 8.0
            min_y = -8.0
            max_y = 8.0

        # Generate random goal positions (x, y) within terrain bounds
        self.goal_positions[envs_idx, 0] = gs_rand_float(
            min_x, max_x, (len(envs_idx),), gs.device
        )
        self.goal_positions[envs_idx, 1] = gs_rand_float(
            min_y, max_y, (len(envs_idx),), gs.device
        )

        # Set goal z height based on terrain height at goal position
        # Goal should be ON the terrain - center of sphere at terrain height + radius
        # so the bottom of the sphere touches the terrain
        goal_radius = 0.1  # Radius of the goal marker sphere
        if self.terrain_heightmap is not None:
            # Get terrain height at goal (x, y) positions
            # Create temporary positions tensor with current x, y and z=0 for height query
            goal_xy_positions = self.goal_positions[envs_idx].clone()
            goal_xy_positions[:, 2] = 0.0  # Set z to 0 for height query
            goal_terrain_heights = self._get_terrain_height_at_positions(
                goal_xy_positions
            )
            # Goal height = terrain height + radius (so sphere sits on terrain)
            # The terrain heightmap is in world coordinates (normalized to center at 0)
            self.goal_positions[envs_idx, 2] = goal_terrain_heights + goal_radius
        else:
            # Flat terrain - goal at ground level + radius
            self.goal_positions[envs_idx, 2] = goal_radius

        # Reset goal reached flags and distances
        self.goal_reached[envs_idx] = False
        self.goal_resample_timer[envs_idx] = 0.0

        # Update goal marker position (only for first environment)
        if self.goal_marker is not None and 0 in envs_idx:
            goal_pos = self.goal_positions[0].cpu().numpy()
            self.goal_marker.set_pos(goal_pos)

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

        # Update terrain height at robot positions (for rewards)
        self.terrain_height_at_robot[:] = self._get_terrain_height_at_positions(
            self.base_pos
        )

        # Update previous goal distances for reward calculation
        if self.use_goal_navigation:
            self.prev_goal_distances[:] = self.goal_distances[:]

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

        # Update goal distances and check if goals are reached
        if self.use_goal_navigation:
            # Calculate distance to goal (only xy distance, ignoring z)
            goal_xy = self.goal_positions[:, :2]
            robot_xy = self.base_pos[:, :2]
            self.goal_distances[:] = torch.norm(goal_xy - robot_xy, dim=1)

            # Check if goals are reached
            self.goal_reached[:] = self.goal_distances < self.goal_reach_distance

            # Resample goals if reached or timer expired
            goal_resample_idx = (
                (
                    (self.goal_reached)
                    | (self.goal_resample_timer >= self.goal_resample_interval)
                )
                .nonzero(as_tuple=False)
                .reshape((-1,))
            )
            if len(goal_resample_idx) > 0:
                self._resample_goals(goal_resample_idx)

            # Update goal resample timer
            self.goal_resample_timer += 1

            # Update goal marker position continuously (only for first environment)
            if self.goal_marker is not None:
                goal_pos = self.goal_positions[0].cpu().numpy()
                self.goal_marker.set_pos(goal_pos)

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

        # Resample goals when resetting
        if self.use_goal_navigation:
            self._resample_goals(envs_idx)
            self.prev_goal_distances[envs_idx] = self.goal_distances[envs_idx]

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))

        # Initialize goals on first reset
        if self.use_goal_navigation:
            self._resample_goals(torch.arange(self.num_envs, device=gs.device))
            # Calculate initial distances
            goal_xy = self.goal_positions[:, :2]
            robot_xy = self.base_pos[:, :2]
            self.goal_distances[:] = torch.norm(goal_xy - robot_xy, dim=1)
            self.prev_goal_distances[:] = self.goal_distances[:]

            # Update goal marker position after reset
            if self.goal_marker is not None:
                goal_pos = self.goal_positions[0].cpu().numpy()
                self.goal_marker.set_pos(goal_pos)

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
        # For random terrain, use relative height (height above terrain)
        # For flat terrain, use absolute height
        if self.terrain_heightmap is not None:
            # Relative height: base height - terrain height
            relative_height = self.base_pos[:, 2] - self.terrain_height_at_robot
            return torch.square(relative_height - self.reward_cfg["base_height_target"])
        else:
            # Flat terrain: use absolute height
            return torch.square(
                self.base_pos[:, 2] - self.reward_cfg["base_height_target"]
            )

    def _reward_goal_distance(self):
        # Reward for reducing distance to goal (negative distance = closer is better)
        # Use exponential reward that increases as distance decreases
        if not self.use_goal_navigation:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Normalize distance (assuming max distance is ~10-15m)
        max_distance = 15.0
        normalized_distance = self.goal_distances / max_distance

        # Exponential reward: higher reward for being closer
        # exp(-distance) gives values between 0 and 1, where 1 is at goal
        return torch.exp(-normalized_distance * 2.0)

    def _reward_goal_velocity(self):
        # Reward for moving towards the goal
        # Positive reward when velocity is in direction of goal
        if not self.use_goal_navigation:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # Direction vector from robot to goal (xy only)
        goal_xy = self.goal_positions[:, :2]
        robot_xy = self.base_pos[:, :2]
        direction_to_goal = goal_xy - robot_xy

        # Normalize direction vector
        distance_to_goal = torch.norm(direction_to_goal, dim=1, keepdim=True)
        # Avoid division by zero
        direction_to_goal_normalized = direction_to_goal / (distance_to_goal + 1e-6)

        # Project velocity onto direction to goal
        # Positive dot product = moving towards goal
        velocity_towards_goal = torch.sum(
            self.base_lin_vel[:, :2] * direction_to_goal_normalized, dim=1
        )

        # Reward positive velocity towards goal
        return torch.clamp(velocity_towards_goal, min=0.0)

    def _reward_goal_reached(self):
        # Large reward when goal is reached
        if not self.use_goal_navigation:
            return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        return self.goal_reached.float()
