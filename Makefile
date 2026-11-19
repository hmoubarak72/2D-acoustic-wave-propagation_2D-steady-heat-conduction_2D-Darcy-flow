PY ?= python

.PHONY: data train bench report test lint app all

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

report:
	$(PY) scripts/make_report.py

test:
	$(PY) -m pytest -q

lint:
	ruff check .

app:
	streamlit run app/Home.py

all: data train bench
