import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def check_dependencies():
    print("Checking dependencies...")
    
    try:
        import pettingzoo
        print("  ✓ pettingzoo")
    except ImportError:
        print("  ✗ pettingzoo (missing)")
        return False
    
    try:
        import torch
        print("  ✓ torch")
    except ImportError:
        print("  ✗ torch (missing)")
        return False
    
    try:
        import numpy
        print("  ✓ numpy")
    except ImportError:
        print("  ✗ numpy (missing)")
        return False
    
    try:
        import pygame
        print("  ✓ pygame")
    except ImportError:
        print("  ✗ pygame (missing)")
        return False
    
    try:
        import pytest
        print("  ✓ pytest")
    except ImportError:
        print("  ✗ pytest (missing)")
        return False
    
    return True


def check_structure():
    print("\nChecking project structure...")
    
    base_path = Path(__file__).parent.parent
    
    required_dirs = [
        'src/engine',
        'src/gui',
        'src/training',
        'src/validation',
        'src/utils',
        'tests',
        'scripts',
        'configs',
        'data/spots',
        'models'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = base_path / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            all_exist = False
    
    return all_exist


def check_config():
    print("\nChecking configuration...")
    
    base_path = Path(__file__).parent.parent
    config_path = base_path / 'configs' / 'training_config.json'
    
    if not config_path.exists():
        print("  ✗ training_config.json (missing)")
        return False
    
    print("  ✓ training_config.json")
    
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_keys = ['game', 'training', 'network', 'paths']
    for key in required_keys:
        if key in config:
            print(f"  ✓ config.{key}")
        else:
            print(f"  ✗ config.{key} (missing)")
            return False
    
    return True


def check_imports():
    print("\nChecking module imports...")
    
    try:
        from src.engine import PokerEngine, PokerValidator
        print("  ✓ src.engine")
    except ImportError as e:
        print(f"  ✗ src.engine ({e})")
        return False
    
    try:
        from src.gui import PokerGUI
        print("  ✓ src.gui")
    except ImportError as e:
        print(f"  ✗ src.gui ({e})")
        return False
    
    try:
        from src.training import DeepCFR, CFRTrainer
        print("  ✓ src.training")
    except ImportError as e:
        print(f"  ✗ src.training ({e})")
        return False
    
    try:
        from src.validation import SpotValidator
        print("  ✓ src.validation")
    except ImportError as e:
        print(f"  ✗ src.validation ({e})")
        return False
    
    try:
        from src.utils import load_config
        print("  ✓ src.utils")
    except ImportError as e:
        print(f"  ✗ src.utils ({e})")
        return False
    
    return True


def main():
    print("="*60)
    print("POKER PROJECT SETUP CHECK")
    print("="*60)
    
    deps_ok = check_dependencies()
    struct_ok = check_structure()
    config_ok = check_config()
    imports_ok = check_imports()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if deps_ok and struct_ok and config_ok and imports_ok:
        print("✓ All checks passed! Project is ready.")
        print("\nNext steps:")
        print("  1. Run quick test: python scripts/quick_test.py")
        print("  2. Run all tests: pytest tests/ -v")
        print("  3. Try GUI: python scripts/play_gui.py")
        print("  4. Run CFR demo: python scripts/demo_cfr.py")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        if not deps_ok:
            print("\nInstall dependencies: pip install -r requirements.txt")
        return 1


if __name__ == "__main__":
    sys.exit(main())
