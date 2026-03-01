import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import PokerEngine
from src.training import DeepCFR, CFRTrainer, ParallelCFRTrainer
import torch


def test_standard_trainer():
    print("\n" + "="*60)
    print("Testing Standard Trainer")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2,
        hidden_layers=[128, 128],
        learning_rate=0.001,
        buffer_size=10000,
        device='cpu'
    )
    
    config = {
        'traversals_per_iteration': 50,
        'batch_size': 32
    }
    
    trainer = CFRTrainer(engine, deep_cfr, config)
    
    start_time = time.time()
    trainer.train_iteration(traversals_per_iteration=50)
    elapsed = time.time() - start_time
    
    print(f"✓ Standard trainer completed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Buffer size: {len(deep_cfr.replay_buffer)}")
    
    engine.close()
    return elapsed


def test_parallel_trainer():
    print("\n" + "="*60)
    print("Testing Parallel Trainer")
    print("="*60)
    
    engine = PokerEngine(num_players=2)
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=2,
        hidden_layers=[128, 128],
        learning_rate=0.001,
        buffer_size=10000,
        device='cpu'
    )
    
    config = {
        'traversals_per_iteration': 50,
        'batch_size': 32
    }
    
    trainer = ParallelCFRTrainer(engine, deep_cfr, config, num_workers=4)
    
    start_time = time.time()
    trainer.train_iteration(traversals_per_iteration=50)
    elapsed = time.time() - start_time
    
    print(f"✓ Parallel trainer completed")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Buffer size: {len(deep_cfr.replay_buffer)}")
    print(f"  Workers: {trainer.num_workers}")
    
    engine.close()
    return elapsed


def test_gpu_available():
    print("\n" + "="*60)
    print("Testing GPU Availability")
    print("="*60)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        engine = PokerEngine(num_players=2)
        observation_size = engine.get_observation_space_size()
        action_size = engine.get_action_space_size()
        
        deep_cfr = DeepCFR(
            observation_size=observation_size,
            action_size=action_size,
            num_players=2,
            hidden_layers=[128, 128],
            learning_rate=0.001,
            buffer_size=10000,
            device='cuda'
        )
        
        print(f"✓ GPU model initialized successfully")
        print(f"  Device: {deep_cfr.device}")
        
        engine.close()
    else:
        print("⚠ No GPU available - GPU training will not be tested")
    
    return cuda_available


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 15 + "PARALLEL TRAINING TEST" + " " * 21 + "║")
    print("╚" + "═" * 58 + "╝")
    
    try:
        gpu_available = test_gpu_available()
        
        standard_time = test_standard_trainer()
        
        parallel_time = test_parallel_trainer()
        
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON")
        print("="*60)
        print(f"Standard trainer: {standard_time:.2f}s")
        print(f"Parallel trainer: {parallel_time:.2f}s")
        speedup = standard_time / parallel_time if parallel_time > 0 else 0
        print(f"Speedup: {speedup:.2f}x")
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
        print("\nRecommendations:")
        if gpu_available:
            print("  🚀 GPU detected! Use train_async.py for maximum performance")
            print("     Command: python3 scripts/train_async.py")
        elif speedup > 2:
            print("  ✓ Good parallel speedup! Use train_parallel.py")
            print("     Command: python3 scripts/train_parallel.py")
        else:
            print("  ⚠ Limited speedup. Consider using more workers or GPU")
        
        print("\nNext steps:")
        print("  1. Run full validation: python3 scripts/validate_train.py")
        print("  2. Start training: python3 scripts/train_parallel.py")
        print("  3. See guide: cat PARALLEL_TRAINING_GUIDE.md")
        print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
