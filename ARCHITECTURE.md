# Architecture Documentation

## System Overview

The poker training engine is built on three main pillars:

1. **Poker Engine** - PettingZoo-based game simulation
2. **Deep CFR** - Neural network-based CFR algorithm
3. **Validation** - Comprehensive testing and spot validation

## Component Details

### Poker Engine (`src/engine/`)

#### PokerEngine
Wraps PettingZoo's Texas Hold'em No Limit environment with a clean API.

Key methods:
- `reset()` - Start new game
- `step(action)` - Execute action
- `get_observation()` - Get current observation
- `get_state()` - Get full game state
- `is_done()` - Check if game finished
- `get_rewards()` - Get final rewards

#### PokerValidator
Validates game states, actions, and poker rules.

Key methods:
- `validate_action()` - Check if action is legal
- `validate_observation()` - Verify observation structure
- `validate_game_state()` - Validate complete state
- `validate_hand_ranking()` - Verify hand evaluation
- `validate_chips_conservation()` - Check chip totals

#### GameState & HandHistory
Track game progression and record hand histories for analysis.

### Deep CFR (`src/training/`)

#### DeepCFR
Main CFR implementation with neural network function approximation.

Components:
- **AdvantageNetwork** - Predicts counterfactual regrets
- **StrategyNetwork** - Learns average strategy
- **ReplayBuffer** - Stores experience for training

Key methods:
- `get_strategy()` - Get current strategy for state
- `get_regret_matching_strategy()` - Compute RM strategy
- `compute_counterfactual_values()` - Get advantage estimates
- `train_advantage_network()` - Update advantage network
- `train_strategy_network()` - Update strategy network

#### CFRTrainer
Orchestrates the training process.

Training loop:
1. Run traversals for each player
2. Collect regrets and strategies
3. Store in replay buffer
4. Train neural networks
5. Checkpoint periodically

#### CFR Utils
Helper functions for CFR computations:
- Regret matching
- Strategy normalization
- Exploitability computation
- Entropy calculation

### GUI (`src/gui/`)

#### PokerGUI
Pygame-based visual interface for manual testing.

Features:
- Visual game representation
- Interactive action buttons
- Real-time state display
- Keyboard shortcuts

Controls:
- Click buttons for actions
- SPACE for random action
- R to reset game
- Q to quit

### Validation (`src/validation/`)

#### SpotValidator
JSON-based validation of specific poker situations.

Features:
- Load spots from JSON files
- Validate expected vs actual state
- Generate validation reports
- Create spots from game states

Spot format:
```json
{
  "id": "spot_name",
  "description": "Spot description",
  "expected_legal_actions": [0, 1, 2],
  "expected_pot_size": 300,
  "notes": "Additional context"
}
```

### Utilities (`src/utils/`)

#### Config Loader
Load and validate configuration files.

#### Training Logger
Log training progress and metrics to JSON files.

## Data Flow

```
Game Start
    ↓
PokerEngine.reset()
    ↓
Get Observation & Legal Actions
    ↓
DeepCFR.get_strategy() → Strategy
    ↓
Sample Action from Strategy
    ↓
PokerEngine.step(action)
    ↓
Compute Regrets
    ↓
ReplayBuffer.add()
    ↓
Train Networks
    ↓
Repeat until convergence
```

## Training Pipeline

### Phase 1: Data Collection
- Run CFR traversals
- Collect (state, action, regret, strategy) tuples
- Store in replay buffer

### Phase 2: Network Training
- Sample batches from replay buffer
- Train advantage network on regrets
- Train strategy network on average strategy

### Phase 3: Evaluation
- Play games with learned strategy
- Compute average rewards
- Measure exploitability

## Testing Strategy

### Unit Tests
- Individual component functionality
- Input/output validation
- Edge case handling

### Integration Tests
- Full game flow
- Multi-game consistency
- State transitions

### Edge Case Tests
- All-in scenarios
- Consecutive folds
- Rapid game cycles
- Zero-sum property

### Spot Validation
- Predefined poker situations
- Expected behavior verification
- Real-world scenario testing

## Configuration

Training parameters in `configs/training_config.json`:

- **Game settings** - Players, chips, blinds
- **Training params** - Iterations, batch size, learning rate
- **Network architecture** - Hidden layers, activation, dropout
- **Paths** - Model storage, data, logs

## Performance

Typical benchmarks:
- 50-100 games/second (random play)
- 10-20 steps average per game
- Zero-sum property maintained
- Memory efficient with replay buffer

## Extensibility

Easy to extend:
- Add new validation rules in `PokerValidator`
- Create custom network architectures
- Implement alternative CFR variants
- Add new spot types for validation
- Integrate different poker variants
