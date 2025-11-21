format:
	ruff format .

train_ppo:
	python3 -m src.go2.go2_train -a ppo

train_sac:
	python3 -m src.go2.go2_train -a sac

eval_ppo:
	python3 -m src.go2.go2_eval -e go2-walking-ppo --ckpt 1000

eval_sac:
	python3 -m src.go2.go2_eval -a sac -e go2-walking-sac --sac_step 57600000

train_and_eval:
	python3 -m src.go2.go2_train
	sleep 20
	python3 -m src.go2.go2_eval -e go2-walking --ckpt 100