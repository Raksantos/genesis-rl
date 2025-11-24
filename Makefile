.DEFAULT_GOAL := help
SHELL := /bin/bash

.PHONY: help format train_ppo train_sac train_td3 train_ddpg eval_ppo eval_sac eval_td3 eval_ddpg

help: ## Mostra este guia de comandos
	@echo "Comandos disponíveis:" && \
	awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  \033[1;36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

format:
	ruff format .

train_ppo: ## Treina PPO com implementacao rsl_rl_lib
	python3 -m src.go2.rsl_lib.ppo_train

train_sac: ## Treina SAC (stable-baselines3)
	python3 -m src.go2.sb3.sac_train

train_td3: ## Treina TD3 (stable-baselines3)
	python3 -m src.go2.sb3.td3_train

train_ddpg: ## Treina DDPG (stable-baselines3)
	python3 -m src.go2.sb3.ddpg_train

train_sac_custom:
	python3 -m src.go2.train_sac_custom

train_td3_custom:
	python3 -m src.go2.train_td3_custom

train_ddpg_custom:
	python3 -m src.go2.train_ddpg_custom

eval_ppo: ## Avalia PPO salvo
	python3 -m src.go2.rsl_lib.ppo_eval -e go2-walking-ppo --ckpt 999

eval_sac: ## Avalia SAC salvo
	python3 -m src.go2.sb3.sac_eval -e go2-sb3-sac

eval_td3: ## Avalia TD3 salvo
	python3 -m src.go2.sb3.td3_eval -e go2-sb3-td3

eval_ddpg: ## Avalia DDPG salvo
	python3 -m src.go2.sb3.ddpg_eval -e go2-sb3-ddpg

eval_sac_custom: ## Avalia SAC salvo (implementacao customizada)
	python3 -m src.go2.eval_sac_custom

eval_td3_custom: ## Avalia TD3 salvo (implementacao customizada)
	python3 -m src.go2.td3_eval_custom

eval_ddpg_custom: ## Avalia DDPG salvo (implementacao customizada)
	python3 -m src.go2.ddpg_eval_custom