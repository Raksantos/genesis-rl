# Genesis - RL for Go2 Locomotion

Repositorio de experimentos de aprendizado por reforco usando Genesis, focado na locomocao do robo quadrupede Go2 e em pipelines customizados de SAC/PPO.

## Estrutura do projeto
```
.
├── src/
│   ├── algorithms/      # SAC, replay buffer, early stopping e runners off-policy
│   ├── configs/         # dataclasses de configuracao do ambiente, algoritmo e seeds
│   ├── go2/             # ambiente Genesis do Go2 e scripts de treinamento/avaliacao
│   ├── helpers/         # metricas de episodios e callbacks auxiliares
│   └── main.py          # cena sandbox simples para carregar assets e inspecionar a simulacao
├── xml/
│   └── go2/             # modelo MJCF e malhas do Go2 usadas pela simulacao
├── logs/                # resultados, checkpoints e registros de execucao
├── pyproject.toml       # dependencias gerenciadas via poetry/pip
└── README.md
```

### Pontos de entrada e arquivos importantes
- `src/go2/train_sac_custom.py`: script principal de treinamento do Go2 com SAC customizado, incluindo avaliacao e checkpoints.
- `src/go2/go2_env.py`: definicao do ambiente Genesis vetorizado, escalas de observacao/recompensa e comandos de locomocao.
- `src/algorithms/off_policy_runner.py`: loop de treino off-policy com buffer, avaliacao periodica e early stopping.
- `src/algorithms/sac.py`: implementacao do agente SAC usado pelo runner.
- `src/configs/`: funcoes `get_cfgs` e dataclasses para montar configuracoes de treino/experimentos.
- `xml/go2/`: arquivos MJCF e meshes do robo.

## Como rodar rapidamente
1) Criar ambiente e instalar deps:
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```
2) Iniciar um treino SAC padrao do Go2:
```
python src/go2/train_sac_custom.py --exp_name go2-custom-sac
```
Use as flags do script para ajustar dispositivo (`--device cpu|cuda`), numero de ambientes e intervalos de avaliacao/checkpoint conforme necessario.
