import json
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, 'r') as f:
        config = json.load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    required_sections = ['game', 'training', 'network', 'paths']
    
    for section in required_sections:
        if section not in config:
            return False
    
    if config['game']['num_players'] < 2:
        return False
    
    if config['training']['num_iterations'] < 1:
        return False
    
    return True
