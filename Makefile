format:
	ruff format .

train:
	python3 -m src.go2.go2_train

eval:
	python3 -m src.go2.go2_eval -e go2-walking --ckpt 100

eval_teleop:
	python3 -m src.go2.go2_eval_teleop -e go2-walking --ckpt 100

train_and_eval:
	python3 -m src.go2.go2_train
	sleep 20
	python3 -m src.go2.go2_eval -e go2-walking --ckpt 100