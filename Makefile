format:
	ruff format .

train:
	python3 -m src.go2.go2_train -B 512

eval:
	python3 -m src.go2.go2_eval -e go2-walking --ckpt 100

eval_teleop:
	python3 -m src.go2.go2_eval_teleop -e go2-walking --ckpt 100