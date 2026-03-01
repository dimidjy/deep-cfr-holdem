import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer
from src.utils import load_config, TrainingLogger


CONFIG_FILE = "configs/training_config.json"


def validate_config():
    print("=" * 60)
    print("VALIDATING TRAINING CONFIGURATION")
    print("=" * 60)
    
    config_path = Path(__file__).parent.parent / CONFIG_FILE
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return None
    
    print(f"✓ Config file found: {config_path}")
    
    try:
        config = load_config(str(config_path))
        print(f"✓ Config loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return None
    
    required_keys = {
        'game': ['num_players'],
        'training': ['num_iterations', 'traversals_per_iteration', 'batch_size', 
                    'learning_rate', 'buffer_size', 'checkpoint_interval'],
        'network': ['hidden_layers'],
        'paths': ['models', 'logs']
    }
    
    for section, keys in required_keys.items():
        if section not in config:
            print(f"❌ Missing config section: {section}")
            return None
        for key in keys:
            if key not in config[section]:
                print(f"❌ Missing config key: {section}.{key}")
                return None
    
    print(f"✓ All required config keys present")
    
    print(f"\nConfiguration values:")
    print(f"  Players: {config['game']['num_players']}")
    print(f"  Iterations: {config['training']['num_iterations']}")
    print(f"  Traversals per iteration: {config['training']['traversals_per_iteration']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Learning rate: {config['training']['learning_rate']}")
    print(f"  Buffer size: {config['training']['buffer_size']}")
    print(f"  Hidden layers: {config['network']['hidden_layers']}")
    
    return config


def validate_directories(config):
    print("\n" + "=" * 60)
    print("VALIDATING DIRECTORIES")
    print("=" * 60)
    
    base_path = Path(__file__).parent.parent
    
    for path_key, path_value in config['paths'].items():
        full_path = base_path / path_value
        if not full_path.exists():
            print(f"⚠ Creating directory: {full_path}")
            full_path.mkdir(parents=True, exist_ok=True)
        else:
            print(f"✓ Directory exists: {full_path}")


def validate_engine(config):
    print("\n" + "=" * 60)
    print("VALIDATING POKER ENGINE")
    print("=" * 60)
    
    try:
        engine = PokerEngine(num_players=config['game']['num_players'])
        print(f"✓ Engine initialized with {config['game']['num_players']} players")
    except Exception as e:
        print(f"❌ Failed to initialize engine: {e}")
        return None
    
    try:
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        print(f"✓ Observation space size: {observation_size}")
        print(f"✓ Action space size: {action_size}")
    except Exception as e:
        print(f"❌ Failed to get space sizes: {e}")
        engine.close()
        return None
    
    try:
        state = engine.reset()
        print(f"✓ Engine reset successful")
        print(f"  Current player: {state['current_player']}")
        print(f"  Legal actions: {state['legal_actions']}")
    except Exception as e:
        print(f"❌ Failed to reset engine: {e}")
        engine.close()
        return None
    
    try:
        if state['legal_actions']:
            action = state['legal_actions'][0]
            result = engine.step(action)
            print(f"✓ Engine step successful")
    except Exception as e:
        print(f"❌ Failed to step engine: {e}")
        engine.close()
        return None
    
    return engine


def validate_deep_cfr(config, engine):
    print("\n" + "=" * 60)
    print("VALIDATING DEEP CFR")
    print("=" * 60)
    
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    try:
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=config['game']['num_players'],
            hidden_layers=config['network']['hidden_layers'],
            learning_rate=config['training']['learning_rate'],
            buffer_size=config['training']['buffer_size']
        )
        print(f"✓ DeepCFR initialized")
    except Exception as e:
        print(f"❌ Failed to initialize DeepCFR: {e}")
        return None
    
    print(f"  Advantage networks: {len(deep_cfr.advantage_networks)}")
    print(f"  Strategy networks: {len(deep_cfr.strategy_networks)}")
    print(f"  Replay buffer capacity: {deep_cfr.replay_buffer.capacity}")
    
    return deep_cfr


def validate_trainer(config, engine, deep_cfr):
    print("\n" + "=" * 60)
    print("VALIDATING CFR TRAINER")
    print("=" * 60)
    
    try:
        trainer = CFRTrainer(engine, deep_cfr, config['training'])
        print(f"✓ CFRTrainer initialized")
    except Exception as e:
        print(f"❌ Failed to initialize CFRTrainer: {e}")
        return None
    
    print(f"  Initial iteration: {trainer.iteration}")
    
    return trainer


def validate_logger(config):
    print("\n" + "=" * 60)
    print("VALIDATING LOGGER")
    print("=" * 60)
    
    try:
        logger = TrainingLogger(config['paths']['logs'])
        print(f"✓ TrainingLogger initialized")
        print(f"  Log directory: {config['paths']['logs']}")
    except Exception as e:
        print(f"❌ Failed to initialize logger: {e}")
        return None
    
    return logger


def run_mini_training_test(trainer, deep_cfr):
    print("\n" + "=" * 60)
    print("RUNNING MINI TRAINING TEST")
    print("=" * 60)
    
    print("Running 1 iteration with 10 traversals...")
    
    try:
        trainer.train_iteration(traversals_per_iteration=10)
        print(f"✓ Training iteration completed")
        print(f"  Iteration: {trainer.iteration}")
        print(f"  Buffer size: {len(deep_cfr.replay_buffer)}")
    except Exception as e:
        print(f"❌ Training iteration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\nRunning evaluation with 5 games...")
    
    try:
        results = trainer.evaluate(num_games=5)
        print(f"✓ Evaluation completed")
        print(f"  Games played: {results['total_games']}")
        for player_idx in range(deep_cfr.num_players):
            avg_reward = results[f'player_{player_idx}_avg_reward']
            print(f"  Player {player_idx} avg reward: {avg_reward:.2f}")
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "TRAIN.PY VALIDATION" + " " * 24 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")
    
    config = validate_config()
    if config is None:
        print("\n❌ VALIDATION FAILED: Config validation failed")
        return 1
    
    validate_directories(config)
    
    engine = validate_engine(config)
    if engine is None:
        print("\n❌ VALIDATION FAILED: Engine validation failed")
        return 1
    
    deep_cfr = validate_deep_cfr(config, engine)
    if deep_cfr is None:
        print("\n❌ VALIDATION FAILED: DeepCFR validation failed")
        engine.close()
        return 1
    
    trainer = validate_trainer(config, engine, deep_cfr)
    if trainer is None:
        print("\n❌ VALIDATION FAILED: Trainer validation failed")
        engine.close()
        return 1
    
    logger = validate_logger(config)
    if logger is None:
        print("\n❌ VALIDATION FAILED: Logger validation failed")
        engine.close()
        return 1
    
    success = run_mini_training_test(trainer, deep_cfr)
    
    engine.close()
    
    if success:
        print("\n" + "=" * 60)
        print("✓ ALL VALIDATIONS PASSED")
        print("=" * 60)
        print("\nYour train.py script is ready to run!")
        print("\nTo start training, run:")
        print("  python3 scripts/train.py")
        print("\nFor a quick demo (5 iterations), run:")
        print("  python3 scripts/demo_cfr.py")
        print("=" * 60)
        return 0
    else:
        print("\n❌ VALIDATION FAILED: Mini training test failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
