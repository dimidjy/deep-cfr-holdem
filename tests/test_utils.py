import pytest
from pathlib import Path
from src.utils import load_config


class TestUtils:
    
    def test_load_config(self):
        config_path = Path(__file__).parent.parent / 'configs' / 'training_config.json'
        
        if config_path.exists():
            config = load_config(str(config_path))
            
            assert 'game' in config
            assert 'training' in config
            assert 'network' in config
            assert 'paths' in config
    
    def test_load_config_nonexistent(self):
        with pytest.raises(FileNotFoundError):
            load_config('nonexistent_config.json')
