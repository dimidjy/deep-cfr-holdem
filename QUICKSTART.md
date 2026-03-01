# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Quick Test

Test the poker engine with random games:

```bash
python scripts/quick_test.py
```

## Run All Tests

```bash
pytest tests/ -v
```

Or use the test runner:

```bash
python scripts/run_tests.py
```

## GUI Testing

Launch the Pygame GUI to manually test the poker engine:

```bash
python scripts/play_gui.py
```

Controls:
- Click buttons to take actions
- Press SPACE for random action
- Press R to reset game
- Press Q to quit

## Validate Spots

Test the engine against predefined poker scenarios:

```bash
python scripts/validate_spots.py
```

## Deep CFR Demo

Run a quick Deep CFR training demo:

```bash
python scripts/demo_cfr.py
```

## Full Training

Train a poker AI model:

```bash
python scripts/train.py
```

Configuration is in `configs/training_config.json`.

## Evaluate Model

After training, evaluate the model:

```bash
python scripts/evaluate_model.py
```

## Play vs AI

Play against the trained AI:

```bash
python scripts/play_vs_ai.py
```
