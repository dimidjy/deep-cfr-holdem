.PHONY: install test quick-test gui demo train evaluate benchmark analyze clean

install:
	pip install -r requirements.txt
	python scripts/check_setup.py

test:
	pytest tests/ -v --tb=short

quick-test:
	python scripts/quick_test.py

gui:
	python scripts/play_gui.py

demo:
	python scripts/demo_cfr.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate_model.py

benchmark:
	python scripts/benchmark.py

analyze:
	python scripts/analyze_engine.py

validate-spots:
	python scripts/validate_spots.py

record-hands:
	python scripts/record_hands.py

play-vs-ai:
	python scripts/play_vs_ai.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf models/*.pt models/*.json
	rm -rf data/logs/*
	rm -rf data/training/*

help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test          - Run all tests"
	@echo "  make quick-test    - Quick engine test"
	@echo "  make gui           - Launch Pygame GUI"
	@echo "  make demo          - Run CFR demo"
	@echo "  make train         - Train full model"
	@echo "  make evaluate      - Evaluate trained model"
	@echo "  make benchmark     - Benchmark engine performance"
	@echo "  make analyze       - Analyze engine statistics"
	@echo "  make validate-spots - Validate poker spots"
	@echo "  make record-hands  - Record hand histories"
	@echo "  make play-vs-ai    - Play against AI"
	@echo "  make clean         - Clean generated files"
