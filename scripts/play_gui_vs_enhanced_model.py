import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui import PokerGUI
from src.training import DeepCFR, ModelAdapter
from src.engine import EnhancedPokerEngine
import torch


def load_model(model_path: str, num_players: int = 2):
    print(f"\nLoading ENHANCED model from: {model_path}")
    
    # Try to detect the correct number of players from the model
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        if 'advantage_networks' in checkpoint and len(checkpoint['advantage_networks']) > 0:
            first_layer_weight = checkpoint['advantage_networks'][0]['network.0.weight']
            model_input_size = first_layer_weight.shape[1]
            
            # Try different player counts to find the right one
            for test_players in [2, 3, 4, 5, 6, 7, 8, 9]:
                test_engine = EnhancedPokerEngine(
                    num_players=test_players,
                    add_position_features=True,
                    add_pot_odds_features=True,
                    add_hand_strength_features=True
                )
                if test_engine.get_observation_space_size() == model_input_size:
                    num_players = test_players
                    print(f"  Auto-detected: {num_players} players (obs size: {model_input_size})")
                    break
    except Exception as e:
        print(f"  Could not auto-detect player count: {e}")
    
    engine = EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True
    )
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
    parser.add_argument('--players', type=int, default=2, help='Number of players (default: 2)')
    
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
    print("Starting GUI - Playing against ENHANCED trained AI")
    print("="*60)
    print("Enhanced features:")
    print("  ✓ Explicit position encoding")
    print("  ✓ Pot odds calculation")
    print("  ✓ Hand strength features")
    print()
    
    # Check if model was trained with more than 2 players
    gui_players = 2  # GUI only supports heads-up
    
    if trained_players != gui_players:
        print(f"⚠️  Model trained with {trained_players} players")
        print(f"   Using adapter to play heads-up (2 players)")
        print(f"   The AI will adapt its strategy accordingly.")
        print()
        
        # Create adapter
        adapter = ModelAdapter(trained_players=trained_players, target_players=gui_players)
        
        # Wrap the model with adapter
        class AdaptedModel:
            def __init__(self, model, adapter):
                self.model = model
                self.adapter = adapter
            
            def get_strategy(self, obs, player_idx, legal_actions):
                # Adapt observation from 2-player to 6-player format
                adapted_obs = self.adapter.adapt_observation(obs)
                # Call the original model with adapted observation
                return self.model.get_strategy(adapted_obs, player_idx, legal_actions)
        
        ai_model = AdaptedModel(deep_cfr, adapter)
    else:
        ai_model = deep_cfr
    
    print("Controls:")
    print("  - Click buttons to take actions")
    print("  - R: Start new hand")
    print("  - V: Show/Hide opponent cards")
    print("  - Q: Quit")
    print("="*60 + "\n")
    
    # Create GUI with enhanced engine
    gui = PokerGUI(num_players=gui_players, ai_model=ai_model)
    
    # Replace the GUI's engine with an enhanced one
    gui.engine = EnhancedPokerEngine(
        num_players=gui_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True
    )
    gui.engine.reset()
    
    gui.run()


if __name__ == "__main__":
    main()
