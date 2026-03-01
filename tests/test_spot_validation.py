import pytest
import json
from pathlib import Path
from src.validation import SpotValidator


class TestSpotValidation:
    
    def test_spot_validator_initialization(self):
        validator = SpotValidator()
        assert validator is not None
        assert validator.spots == []
    
    def test_load_spots_from_file(self):
        spots_file = Path(__file__).parent.parent / 'data' / 'spots' / 'sample_spots.json'
        
        if spots_file.exists():
            validator = SpotValidator(str(spots_file))
            assert len(validator.spots) > 0
    
    def test_validate_spot_legal_actions_match(self):
        validator = SpotValidator()
        
        spot = {
            'id': 'test_spot',
            'expected_legal_actions': [0, 1, 2]
        }
        
        game_state = {
            'legal_actions': [0, 1, 2]
        }
        
        result = validator.validate_spot(spot, game_state)
        assert result['passed'] is True
        assert len(result['errors']) == 0
    
    def test_validate_spot_legal_actions_mismatch(self):
        validator = SpotValidator()
        
        spot = {
            'id': 'test_spot',
            'expected_legal_actions': [0, 1, 2]
        }
        
        game_state = {
            'legal_actions': [0, 1]
        }
        
        result = validator.validate_spot(spot, game_state)
        assert result['passed'] is False
        assert len(result['errors']) > 0
    
    def test_validate_spot_pot_size_match(self):
        validator = SpotValidator()
        
        spot = {
            'id': 'test_spot',
            'expected_pot_size': 500
        }
        
        game_state = {
            'pot_size': 500
        }
        
        result = validator.validate_spot(spot, game_state)
        assert result['passed'] is True
    
    def test_validate_spot_pot_size_mismatch(self):
        validator = SpotValidator()
        
        spot = {
            'id': 'test_spot',
            'expected_pot_size': 500
        }
        
        game_state = {
            'pot_size': 600
        }
        
        result = validator.validate_spot(spot, game_state)
        assert result['passed'] is False
    
    def test_validate_spot_current_player_match(self):
        validator = SpotValidator()
        
        spot = {
            'id': 'test_spot',
            'expected_current_player': 'player_0'
        }
        
        game_state = {
            'current_player': 'player_0'
        }
        
        result = validator.validate_spot(spot, game_state)
        assert result['passed'] is True
    
    def test_validate_all_spots(self):
        validator = SpotValidator()
        validator.spots = [
            {'id': 'spot1', 'expected_legal_actions': [0, 1]},
            {'id': 'spot2', 'expected_legal_actions': [0, 1, 2]}
        ]
        
        game_states = [
            {'legal_actions': [0, 1]},
            {'legal_actions': [0, 1, 2]}
        ]
        
        results = validator.validate_all_spots(game_states)
        
        assert results['total_spots'] == 2
        assert results['passed'] == 2
        assert results['failed'] == 0
        assert results['pass_rate'] == 1.0
    
    def test_create_spot_from_state(self):
        validator = SpotValidator()
        
        game_state = {
            'legal_actions': [0, 1, 2],
            'current_player': 'player_0',
            'pot_size': 300
        }
        
        spot = validator.create_spot_from_state(game_state, 'test_spot', 'Test description')
        
        assert spot['id'] == 'test_spot'
        assert spot['description'] == 'Test description'
        assert spot['expected_legal_actions'] == [0, 1, 2]
        assert spot['expected_current_player'] == 'player_0'
    
    def test_generate_sample_spots(self):
        spots = SpotValidator.generate_sample_spots()
        
        assert len(spots) > 0
        assert all('id' in spot for spot in spots)
        assert all('description' in spot for spot in spots)
