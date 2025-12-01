import argparse
import torch
import genesis as gs

from src.go2 import Go2Env
from src.configs import get_cfgs, set_global_seed


def main():
    parser = argparse.ArgumentParser(description="Visualiza o ambiente Go2")
    parser.add_argument(
        "--device", type=str, default="cuda:0", choices=["cuda:0", "cpu"]
    )
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument(
        "--num_steps", type=int, default=2000, help="Número de passos para visualizar"
    )
    args = parser.parse_args()

    # Inicializa o Genesis
    set_global_seed()

    # Carrega configurações
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()

    # Cria o ambiente com viewer habilitado
    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Reset do ambiente
    obs, _ = env.reset()
    print("Ambiente inicializado. Visualizador aberto.")
    print("Robô caindo pela física...")
    print(f"Pressione Ctrl+C para sair.")

    # Loop de visualização - apenas física, sem ações e sem resets automáticos
    try:
        for step in range(args.num_steps):
            # Usa diretamente a simulação física sem passar pelo step() do ambiente
            # Isso evita resets automáticos quando o robô cai
            env.scene.step()

            # Print progresso a cada 100 passos
            if (step + 1) % 100 == 0:
                # Obtém posição do robô diretamente
                base_pos = env.robot.get_pos()
                base_vel = env.robot.get_vel()
                vel_magnitude = torch.norm(base_vel[0]).item()
                print(
                    f"Passo {step + 1}/{args.num_steps} - Posição Z: {base_pos[0, 2]:.3f}m - Velocidade: {vel_magnitude:.3f}m/s"
                )

    except KeyboardInterrupt:
        print("\nVisualização interrompida pelo usuário.")

    print("Fechando ambiente...")


if __name__ == "__main__":
    main()
