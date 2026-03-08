import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui import PokerGUI
from src.training import DeepCFR, ModelAdapter
from src.engine import EnhancedPokerEngine
import torch


def _make_engine(num_players: int) -> EnhancedPokerEngine:
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
    )


def load_model(model_path: str, num_players: int = 6):
    print(f"\nLoading ENHANCED model from: {model_path}")

    # Auto-detect player count from checkpoint input size
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'advantage_networks' in checkpoint and len(checkpoint['advantage_networks']) > 0:
            first_layer_weight = checkpoint['advantage_networks'][0]['network.0.weight']
            model_input_size = first_layer_weight.shape[1]

            for test_players in [2, 3, 4, 5, 6, 7, 8, 9]:
                if _make_engine(test_players).get_observation_space_size() == model_input_size:
                    num_players = test_players
                    print(f"  Auto-detected: {num_players} players (obs size: {model_input_size})")
                    break
            else:
                print(f"  WARNING: could not match obs size {model_input_size}, using {num_players} players")
    except Exception as e:
        print(f"  Could not auto-detect player count: {e}")

    engine = _make_engine(num_players)
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()

    print(f"  Enhanced observation size: {observation_size}")
    print(f"  Number of players: {num_players}")

    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=num_players,
        hidden_layers=[512, 512, 256],
        learning_rate=0.001,
        buffer_size=100000,
        device='cpu'
    )

    try:
        deep_cfr.load_models(model_path)
        print(f"✓ Enhanced model loaded successfully")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Players: {num_players}")
        return deep_cfr, engine, num_players
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None


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
        model_type = "enhanced" if "enhanced" in model_file.name else "base"
        print(f"  {i}. {model_file.name} ({size:.1f} MB) [{model_type}]")
    
    return model_files


def main():
    parser = argparse.ArgumentParser(description='Play poker against a trained ENHANCED AI model')
    parser.add_argument('--model', type=str, help='Path to enhanced model file')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--players', type=int, default=6, help='Number of players fallback if auto-detect fails (default: 6)')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        print("Please specify a model with --model or use --list to see available models")
        print("\nExample:")
        print("  python3 scripts/play_gui_vs_enhanced_model.py --model models/final_model_enhanced.pt")
        print("  python3 scripts/play_gui_vs_enhanced_model.py --list")
        print("\n⚠️  This script is for ENHANCED models only!")
        print("    For base models, use: python3 scripts/play_gui_vs_model.py")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("\nUse --list to see available models")
        return
    
    deep_cfr, engine, trained_players = load_model(str(model_path), args.players)
    if deep_cfr is None or engine is None:
        print("\n⚠️  If you're trying to load a BASE model, use:")
        print("    python3 scripts/play_gui_vs_model.py --model", args.model)
        return

    print("\n" + "="*60)
    print(f"Starting GUI - You vs {trained_players - 1} AI opponents ({trained_players}-player game)")
    print("="*60)
    print("Controls:")
    print("  - Click buttons to take actions")
    print("  - R: Start new hand")
    print("  - V: Show/Hide opponent cards")
    print("  - Q: Quit")
    print("="*60 + "\n")

    gui = PokerGUI(num_players=trained_players, ai_model=deep_cfr)
    gui.engine = engine
    gui.engine.reset()
    gui.run()


if __name__ == "__main__":
    main()
