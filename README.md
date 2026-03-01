# Poker Training Engine

Deep CFR-based poker AI training system using PettingZoo Texas Hold'em No Limit.

## Features

- PettingZoo Texas Hold'em No Limit environment wrapper
- Deep CFR implementation with neural network approximation
- Comprehensive test suite with 50+ tests
- Pygame GUI for manual testing
- JSON-based spot validation system
- Training logging and checkpointing
- Model evaluation and comparison tools
- Interactive Jupyter notebooks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Or use the setup script:

```bash
bash scripts/setup.sh
```

Or use Make:

```bash
make install
```

### Verify Setup

```bash
python scripts/check_setup.py
```

### Quick Test

```bash
python scripts/quick_test.py
# or
make quick-test
```

## Usage

### Run All Tests

```bash
pytest tests/ -v
# or
make test
```

### Launch GUI

```bash
python scripts/play_gui.py
# or
make gui
```

### Run CFR Demo

```bash
python scripts/demo_cfr.py
# or
make demo
```

### Train Model

```bash
python scripts/train.py
# or
make train
```

### Evaluate Model

```bash
python scripts/evaluate_model.py
# or
make evaluate
```

### Additional Tools

```bash
make benchmark          # Performance benchmarking
make analyze           # Statistical analysis
make validate-spots    # Validate poker spots
make record-hands      # Record hand histories
make play-vs-ai        # Play against trained AI
```

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for detailed structure.

See [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details.

See [QUICKSTART.md](QUICKSTART.md) for quick reference.

## Configuration

Training configuration in `configs/training_config.json`:

- `training_config.json` - Full training (1000 iterations)
- `demo_config.json` - Quick demo (10 iterations)
- `fast_config.json` - Fast test (5 iterations)

## Testing

The project includes comprehensive tests:

- `test_engine.py` - Engine functionality
- `test_validator.py` - Validation logic
- `test_integration.py` - Full game flow
- `test_edge_cases.py` - Corner cases
- `test_deep_cfr.py` - CFR networks
- `test_cfr_trainer.py` - Training pipeline
- `test_spot_validation.py` - Spot validation
- `test_cfr_utils.py` - CFR utilities
- `test_game_state.py` - State management

## Development

### Interactive Exploration

Use the Jupyter notebook:

```bash
jupyter notebook notebooks/explore_engine.ipynb
```

### Adding New Spots

Edit `data/spots/sample_spots.json` or create new spot files.

### Custom Training

Modify `configs/training_config.json` or create custom config files.
