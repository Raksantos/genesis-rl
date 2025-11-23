format:
	ruff format .

train_ppo:
	python3 -m src.go2.ppo_train

train_sac:
	python3 -m src.go2.sac_train

train_td3:
	python3 -m src.go2.td3_train

train_ddpg:
	python3 -m src.go2.ddpg_train

eval_ppo:
	python3 -m src.go2.ppo_eval -e go2-walking-ppo --ckpt 1100

eval_sac:
	python3 -m src.go2.sac_eval -a sac -e go2-walking-sac --sac_step 57600000

eval_td3:
	python3 -m src.go2.td3_eval -e go2-sb3-td3

eval_ddpg:
	python3 -m src.go2.ddpg_eval -e go2-sb3-ddpg