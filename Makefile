.PHONY: install test models equity train selfplay tournament spots webui gui clean help

MODEL ?= models/best/checkpoint_iter_4000.pt
HANDS ?= 10000
EQUITY ?= models/equity_model_v3.pt

install:
	pip install -r requirements.txt

test:
	python3 -m pytest tests/ -q

models:
	bash scripts/download_models.sh

equity:
	python3 scripts/train_equity_model.py --samples 10000000 --epochs 80 --output $(EQUITY)

train:
	@test -f $(EQUITY) || { echo "Missing $(EQUITY) - run 'make equity' first"; exit 1; }
	python3 scripts/train_gpu.py --config configs/training_config.json \
		--equity-model $(EQUITY)

selfplay:
	python3 scripts/analysis/self_play.py --model $(MODEL) --hands $(HANDS)

tournament:
	python3 scripts/tournament.py --models-dir models/best --output-dir tournament/latest

spots:
	python3 scripts/inspect_model_spots.py --model $(MODEL)

webui:
	python3 -m webui.server --host 127.0.0.1

gui:
	python3 scripts/play_gui_vs_enhanced_model.py --model $(MODEL)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "make install     install dependencies"
	@echo "make test        run the test suite"
	@echo "make models      download the published checkpoint + equity model"
	@echo "make equity      train the neural equity model (needed by train)"
	@echo "make train       Deep CFR training (configs/training_config.json)"
	@echo "make selfplay    6-max self-play stats for MODEL (HANDS hands)"
	@echo "make tournament  spot / HH / head-to-head evaluation of models/best"
	@echo "make spots       raw vs pipeline decisions on hand-picked spots"
	@echo "make webui       job runner and spot explorer on localhost:8000"
	@echo "make gui         play against MODEL in a pygame window"
