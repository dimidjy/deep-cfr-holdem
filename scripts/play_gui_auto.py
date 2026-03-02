import sys
import os
import argparse
from pathlib import Path
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def detect_model_type(model_path: str) -> str:
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'advantage_networks' in checkpoint and len(checkpoint['advantage_networks']) > 0:
            first_layer_weight = checkpoint['advantage_networks'][0]['network.0.weight']
            input_size = first_layer_weight.shape[1]
            
            if input_size == 54:
                return 'base'
            elif input_size == 67:
                return 'enhanced'
            else:
                return f'unknown (input_size={input_size})'
        
        return 'unknown'
    except Exception as e:
        return f'error: {e}'


def main():
    parser = argparse.ArgumentParser(description='Auto-detect and play against trained AI model')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--list', action='store_true', help='List available models with types')
    parser.add_argument('--players', type=int, default=2, help='Number of players (default: 2)')
    
    args = parser.parse_args()
    
    if args.list:
        models_dir = Path("models")
        if not models_dir.exists():
            print("No models directory found.")
            return
        
        model_files = list(models_dir.glob("*.pt"))
        if not model_files:
            print("No model files found in models/")
            return
        
        print("\nAvailable models:")
        print("-" * 80)
        for model_file in sorted(model_files):
            size = model_file.stat().st_size / (1024 * 1024)
            model_type = detect_model_type(str(model_file))
            print(f"{model_file.name:40s} {size:6.1f} MB  [{model_type}]")
        print("-" * 80)
        return
    
    if not args.model:
        print("Please specify a model with --model or use --list to see available models")
        print("\nExample:")
        print("  python3 scripts/play_gui_auto.py --model models/final_model.pt")
        print("  python3 scripts/play_gui_auto.py --list")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        return
    
    print(f"\nDetecting model type...")
    model_type = detect_model_type(str(model_path))
    print(f"Model type: {model_type}")
    
    if model_type == 'base':
        print("\n→ Launching with BASE engine...")
        os.system(f"python3 scripts/play_gui_vs_model.py --model {args.model} --players {args.players}")
    elif model_type == 'enhanced':
        print("\n→ Launching with ENHANCED engine...")
        os.system(f"python3 scripts/play_gui_vs_enhanced_model.py --model {args.model} --players {args.players}")
    else:
        print(f"\n❌ Unknown model type: {model_type}")
        print("\nPlease use the appropriate script:")
        print("  - For base models: python3 scripts/play_gui_vs_model.py")
        print("  - For enhanced models: python3 scripts/play_gui_vs_enhanced_model.py")


if __name__ == "__main__":
    main()
