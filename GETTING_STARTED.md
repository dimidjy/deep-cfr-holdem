# Getting Started

## Installation (2 minutes)

```bash
pip install -r requirements.txt
python scripts/check_setup.py
```

## Verify Everything Works (1 minute)

```bash
python scripts/quick_test.py
```

Expected output: 10 random games with statistics showing the engine works correctly.

## Try the GUI (Interactive)

```bash
python scripts/play_gui.py
```

- Click action buttons to play
- Press SPACE for random action
- Press R to reset
- Press Q to quit

## Run Full Test Suite (2-3 minutes)

```bash
pytest tests/ -v
```

Expected: 50+ tests all passing.

## Quick CFR Demo (2-3 minutes)

```bash
python scripts/demo_cfr.py
```

Runs 5 training iterations with 50 traversals each. Shows Deep CFR working.

## Analyze Engine Performance (1 minute)

```bash
python scripts/analyze_engine.py
```

Shows detailed statistics about game length, action distribution, rewards, etc.

## Benchmark Performance (30 seconds)

```bash
python scripts/benchmark.py
```

Measures games/second and steps/second.

## Validate Poker Spots (30 seconds)

```bash
python scripts/validate_spots.py
```

Tests engine against predefined poker scenarios.

## Compare Strategies (1 minute)

```bash
python scripts/compare_strategies.py
```

Compares random, aggressive, and conservative strategies.

## Full Training (Hours)

```bash
python scripts/train.py
```

Uses `configs/training_config.json` (1000 iterations).

For faster testing:
```bash
python scripts/train.py  # Edit to use demo_config.json
```

## After Training

### Evaluate Model
```bash
python scripts/evaluate_model.py
```

### Play Against AI
```bash
python scripts/play_vs_ai.py
```

## Using Make Commands

```bash
make install          # Install dependencies
make test            # Run all tests
make quick-test      # Quick verification
make gui             # Launch GUI
make demo            # CFR demo
make train           # Train model
make evaluate        # Evaluate model
make benchmark       # Performance test
make analyze         # Statistical analysis
make validate-spots  # Spot validation
make play-vs-ai      # Play against AI
make clean           # Clean generated files
make help            # Show all commands
```

## Troubleshooting

### Import Errors
Make sure you're running scripts from the project root or they handle paths correctly.

### Pygame Issues
If GUI doesn't work, check Pygame installation:
```bash
pip install --upgrade pygame
```

### CUDA/GPU
To use GPU for training, install PyTorch with CUDA:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Then modify training to use `device='cuda'`.

### Memory Issues
Reduce buffer size in config if running out of memory:
```json
"buffer_size": 10000
```

## What to Explore

1. **Interactive Notebook**: `notebooks/explore_engine.ipynb`
2. **Configuration Files**: `configs/` directory
3. **Sample Spots**: `data/spots/` directory
4. **Test Examples**: `tests/` directory
5. **Usage Examples**: `USAGE_EXAMPLES.md`

## Project Health

- ✓ Zero linter errors
- ✓ 50+ passing tests
- ✓ Comprehensive documentation
- ✓ Multiple entry points
- ✓ Ready for production use

## Development Workflow

1. Make changes to source code
2. Run tests: `make test`
3. Verify with quick test: `make quick-test`
4. Test in GUI: `make gui`
5. Run full analysis: `make analyze`

## Extending the Project

### Add New Validation Rules
Edit `src/engine/validator.py` and add methods.

### Create Custom Spots
Edit `data/spots/sample_spots.json` or create new JSON files.

### Modify Network Architecture
Edit `configs/training_config.json` hidden_layers.

### Add New Strategies
Create functions in `scripts/compare_strategies.py`.

### Custom Training Loop
Extend `src/training/cfr_trainer.py` with new methods.
