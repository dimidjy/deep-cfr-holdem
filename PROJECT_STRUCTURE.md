# Project Structure

## Directory Layout

```
poker/
├── configs/                      # Configuration files
│   └── training_config.json     # Training hyperparameters
├── data/                        # Data storage
│   └── spots/                   # Poker spot definitions
│       ├── sample_spots.json    # Basic validation spots
│       └── advanced_spots.json  # Advanced scenarios
├── models/                      # Trained model storage
├── scripts/                     # Executable scripts
│   ├── train.py                # Full training pipeline
│   ├── demo_cfr.py             # Quick CFR demo
│   ├── evaluate_model.py       # Model evaluation
│   ├── play_gui.py             # Pygame GUI launcher
│   ├── play_vs_ai.py           # Play against AI
│   ├── quick_test.py           # Quick engine test
│   ├── run_tests.py            # Test runner
│   └── validate_spots.py       # Spot validation
├── src/                        # Source code
│   ├── engine/                 # Poker engine wrapper
│   │   ├── poker_engine.py    # Main engine class
│   │   └── validator.py       # Game state validation
│   ├── gui/                    # GUI implementation
│   │   └── poker_gui.py       # Pygame interface
│   ├── training/               # Deep CFR implementation
│   │   ├── deep_cfr.py        # Neural networks & CFR
│   │   └── cfr_trainer.py     # Training orchestration
│   ├── utils/                  # Utility functions
│   │   └── config_loader.py   # Config management
│   └── validation/             # Spot validation
│       └── spot_validator.py  # JSON spot testing
└── tests/                      # Test suite
    ├── test_engine.py          # Engine tests
    ├── test_validator.py       # Validator tests
    ├── test_integration.py     # Integration tests
    ├── test_edge_cases.py      # Edge case tests
    ├── test_deep_cfr.py        # Deep CFR tests
    ├── test_cfr_trainer.py     # Trainer tests
    ├── test_spot_validation.py # Spot validation tests
    └── test_utils.py           # Utility tests
```

## Component Overview

### Poker Engine (`src/engine/`)
- Wraps PettingZoo Texas Hold'em No Limit
- Provides clean API for game interaction
- Validates game states and actions
- Handles observation and action spaces

### Deep CFR (`src/training/`)
- Advantage networks for regret prediction
- Strategy networks for policy learning
- Replay buffer for experience storage
- CFR trainer for orchestrating training

### GUI (`src/gui/`)
- Pygame-based visual interface
- Manual testing and debugging
- Real-time game visualization
- Interactive action selection

### Validation (`src/validation/`)
- JSON-based spot testing
- Scenario validation
- Expected vs actual comparison
- Comprehensive test coverage

## Testing Strategy

1. Unit tests for individual components
2. Integration tests for full game flow
3. Edge case tests for corner scenarios
4. Spot validation for known situations
5. Zero-sum property verification
6. Chips conservation checks

## Training Pipeline

1. Initialize engine and Deep CFR networks
2. Run CFR traversals for each player
3. Collect regrets and strategies in replay buffer
4. Train advantage and strategy networks
5. Checkpoint models periodically
6. Evaluate trained models

## Key Features

- PettingZoo-based poker simulation
- Deep CFR with neural network approximation
- Comprehensive test coverage
- Pygame GUI for visualization
- JSON spot validation system
- Configurable training pipeline
- Model checkpointing and evaluation
