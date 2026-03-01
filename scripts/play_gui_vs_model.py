import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui import PokerGUI
from src.training import DeepCFR
from src.engine import PokerEngine
import torch


def load_model(model_path: str, num_players: int = 2):
    print(f"\nLoading model from: {model_path}")
    
    engine = PokerEngine(num_players=num_players)
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()
    
    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=num_players,
        hidden_layers=[256, 256, 128],
        learning_rate=0.001,
        buffer_size=100000,
        device='cpu'
    )
    
    try:
        deep_cfr.load_models(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Players: {num_players}")
        return deep_cfr
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def list_available_models():
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found.")
        return []
    
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("No model files found in models/")
        return []
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files, 1):
        size = model_file.stat().st_size / (1024 * 1024)
        print(f"  {i}. {model_file.name} ({size:.1f} MB)")
    
    return model_files


def main():
    parser = argparse.ArgumentParser(description='Play poker against a trained AI model')
    parser.add_argument('--model', type=str, help='Path to model file (e.g., models/final_model.pt)')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--players', type=int, default=2, help='Number of players (default: 2)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        print("Please specify a model with --model or use --list to see available models")
        print("\nExample:")
        print("  python3 scripts/play_gui_vs_model.py --model models/final_model.pt")
        print("  python3 scripts/play_gui_vs_model.py --list")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("\nUse --list to see available models")
        return
    
    deep_cfr = load_model(str(model_path), args.players)
    if deep_cfr is None:
        return
    
    print("\n" + "="*60)
    print("Starting GUI - Playing against trained AI")
    print("="*60)
    print("Controls:")
    print("  - Click buttons to take actions")
    print("  - R: Start new hand")
    print("  - V: Show/Hide opponent cards")
    print("  - Q: Quit")
    print("="*60)
    print("\nNote: AI strength depends on training iterations.")
    print("Early checkpoints may make suboptimal plays (e.g., folding when checking is free).")
    print("Use models with 500+ iterations for better play.")
    print("="*60 + "\n")
    
    gui = PokerGUI(num_players=args.players, ai_model=deep_cfr)
    gui.run()


if __name__ == "__main__":
    main()
