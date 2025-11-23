format:
	ruff format .

train_ppo:
	python3 -m src.go2.rsl_lib.ppo_train

train_sac:
	python3 -m src.go2.sb3.sac_train

train_td3:
	python3 -m src.go2.sb3.td3_train

train_ddpg:
	python3 -m src.go2.sb3.ddpg_train

eval_ppo:
	python3 -m src.go2.rsl_lib.ppo_eval -e go2-walking-ppo --ckpt 1100

eval_sac:
	python3 -m src.go2.sb3.sac_eval -e go2-sb3-sac

eval_td3:
	python3 -m src.go2.sb3.td3_eval -e go2-sb3-td3

eval_ddpg:
	python3 -m src.go2.sb3.ddpg_eval -e go2-sb3-ddpg