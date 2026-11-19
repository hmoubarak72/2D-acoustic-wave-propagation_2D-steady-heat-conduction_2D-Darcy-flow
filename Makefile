PY ?= python

.PHONY: data train bench data-steady train-steady pinn bench-steady report test lint app all

data:            ## generate datasets (about 1 min)
	$(PY) scripts/generate_data.py

train:           ## train FNO and U-Net on CPU (about 45 min, in parallel)
	$(PY) scripts/train.py --arch fno --epochs 60 --threads 2 & \
	$(PY) scripts/train.py --arch unet --epochs 90 --threads 2 & \
	wait

bench:           ## calibrate the veto, run planted failures, time models, write results/ and docs
	$(PY) scripts/benchmark.py
	$(PY) scripts/scaling.py
	$(PY) scripts/make_report.py

data-steady:     ## Case B (heat) and C (Darcy) datasets (about 1 min)
	$(PY) scripts/generate_data_steady.py

train-steady:    ## FNO, U-Net, physics-loss FNO and 100-sample variants for B and C (about 40 min, in parallel)
	$(PY) scripts/train_steady.py --case case_b --arch fno --name fno --epochs 45 --max-minutes 40 & \
	$(PY) scripts/train_steady.py --case case_b --arch unet --name unet --epochs 45 --max-minutes 40 & \
	$(PY) scripts/train_steady.py --case case_c --arch fno --name fno --epochs 45 --max-minutes 40 & \
	$(PY) scripts/train_steady.py --case case_c --arch unet --name unet --epochs 45 --max-minutes 40 & \
	$(PY) scripts/train_steady.py --case case_b --arch fno --name fno_pino --physics 0.2 --epochs 45 --max-minutes 40 & \
	wait
	$(PY) scripts/train_steady.py --case case_b --arch fno --name fno_n100 --n-train 100 --epochs 200 --max-minutes 20 & \
	$(PY) scripts/train_steady.py --case case_b --arch fno --name fno_pino_n100 --n-train 100 --physics 0.2 --epochs 200 --max-minutes 20 & \
	wait

pinn:            ## classic PINN baseline on 3 Case B test instances
	$(PY) scripts/pinn_case_b.py --instances 3 --iters 5000

bench-steady:    ## calibrate + evaluate the veto for B and C, time solver vs surrogates
	$(PY) scripts/benchmark_steady.py --case case_b
	$(PY) scripts/benchmark_steady.py --case case_c
	$(PY) scripts/make_report.py

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

app:
	streamlit run app/Home.py

all: data train bench data-steady train-steady pinn bench-steady
