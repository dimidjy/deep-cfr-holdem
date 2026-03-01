# Usage Examples

## Basic Engine Usage

```python
from src.engine import PokerEngine
import numpy as np

engine = PokerEngine(num_players=2)
state = engine.reset()

while not engine.is_done():
    legal_actions = engine._get_legal_actions()
    if not legal_actions:
        break
    
    action = np.random.choice(legal_actions)
    state = engine.step(action)

rewards = engine.get_rewards()
print(f"Final rewards: {rewards}")

engine.close()
```

## Using Deep CFR

```python
from src.engine import PokerEngine
from src.training import DeepCFR

engine = PokerEngine(num_players=2)
observation_size = engine.get_observation_space_size()
action_size = engine.get_action_space_size()

deep_cfr = DeepCFR(
    observation_size=observation_size,
    action_size=action_size,
    num_players=2,
    hidden_layers=[256, 256, 128]
)

engine.reset()
observation = engine.get_observation()
legal_actions = engine._get_legal_actions()

strategy = deep_cfr.get_strategy(observation, player_idx=0, legal_actions=legal_actions)
print(f"Strategy: {strategy}")

engine.close()
```

## Training a Model

```python
from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer
from src.utils import load_config

config = load_config('configs/demo_config.json')

engine = PokerEngine(num_players=2)
observation_size = engine.get_observation_space_size()
action_size = engine.get_action_space_size()

deep_cfr = DeepCFR(
    observation_size=observation_size,
    action_size=action_size,
    num_players=2,
    hidden_layers=config['network']['hidden_layers'],
    learning_rate=config['training']['learning_rate']
)

trainer = CFRTrainer(engine, deep_cfr, config['training'])

trainer.train(
    num_iterations=10,
    checkpoint_interval=5
)

trainer.save_checkpoint('models/my_model.pt')

engine.close()
```

## Validating Spots

```python
from src.validation import SpotValidator
from src.engine import PokerEngine

validator = SpotValidator('data/spots/sample_spots.json')

engine = PokerEngine(num_players=2)
game_states = []

for _ in range(len(validator.spots)):
    engine.reset()
    state = engine.get_state()
    game_states.append(state)

results = validator.validate_all_spots(game_states)

print(f"Pass rate: {results['pass_rate']*100:.1f}%")

engine.close()
```

## Recording Hand Histories

```python
from src.engine import PokerEngine, HandHistory
import numpy as np

engine = PokerEngine(num_players=2)
hand_history = HandHistory()

for _ in range(10):
    hand_history.start_hand()
    
    state = engine.reset()
    
    while not engine.is_done():
        legal_actions = engine._get_legal_actions()
        if not legal_actions:
            break
        
        action = np.random.choice(legal_actions)
        current_player = state['current_player']
        
        hand_history.add_action(action, current_player, state)
        state = engine.step(action)
    
    rewards = engine.get_rewards()
    hand_history.end_hand(rewards)

hand_history.save_to_file('data/my_hands.json')

engine.close()
```

## Strategy Analysis

```python
from src.analysis import StrategyAnalyzer
from src.engine import PokerEngine
import numpy as np

analyzer = StrategyAnalyzer()
engine = PokerEngine(num_players=2)

for _ in range(100):
    engine.reset()
    
    while not engine.is_done():
        current_agent = engine.current_agent
        legal_actions = engine._get_legal_actions()
        
        if not legal_actions:
            break
        
        action = np.random.choice(legal_actions)
        analyzer.record_action(action, current_agent)
        
        engine.step(action)

summary = analyzer.get_summary('player_0')
print(f"VPIP: {summary['vpip']*100:.1f}%")
print(f"PFR: {summary['pfr']*100:.1f}%")
print(f"Aggression Factor: {summary['aggression_factor']:.2f}")

engine.close()
```

## Loading and Evaluating a Model

```python
from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer

engine = PokerEngine(num_players=2)

deep_cfr = DeepCFR(
    observation_size=engine.get_observation_space_size(),
    action_size=engine.get_action_space_size(),
    num_players=2
)

deep_cfr.load_models('models/final_model.pt')

trainer = CFRTrainer(engine, deep_cfr)
results = trainer.evaluate(num_games=100)

print(f"Player 0 avg reward: {results['player_0_avg_reward']:.2f}")
print(f"Player 1 avg reward: {results['player_1_avg_reward']:.2f}")

engine.close()
```

## Custom Configuration

```python
from src.utils import load_config

config = {
    'game': {
        'num_players': 2,
        'starting_chips': 10000
    },
    'training': {
        'num_iterations': 50,
        'batch_size': 64,
        'learning_rate': 0.001,
        'buffer_size': 50000,
        'traversals_per_iteration': 100
    },
    'network': {
        'hidden_layers': [256, 256, 128],
        'activation': 'relu',
        'dropout': 0.1
    }
}
```

## Validation with Custom Spots

```python
from src.validation import SpotValidator

validator = SpotValidator()

custom_spot = {
    'id': 'my_spot',
    'description': 'Custom test spot',
    'expected_legal_actions': [0, 1, 2],
    'expected_current_player': 'player_0'
}

game_state = {
    'legal_actions': [0, 1, 2],
    'current_player': 'player_0'
}

result = validator.validate_spot(custom_spot, game_state)

if result['passed']:
    print("Spot validation passed!")
else:
    print(f"Errors: {result['errors']}")
```

## Creating Spots from Game States

```python
from src.engine import PokerEngine
from src.validation import SpotValidator

engine = PokerEngine(num_players=2)
validator = SpotValidator()

engine.reset()
state = engine.get_state()

spot = validator.create_spot_from_state(
    state,
    spot_id='custom_spot',
    description='Generated from game state'
)

spots = [spot]
validator.save_spots('data/spots/my_spots.json', spots)

engine.close()
```
