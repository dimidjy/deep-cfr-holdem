import json
from typing import Dict, List, Any, Optional
from pathlib import Path


class SpotValidator:
    
    def __init__(self, spots_file: Optional[str] = None):
        self.spots_file = spots_file
        self.spots = []
        
        if spots_file:
            self.load_spots(spots_file)
    
    def load_spots(self, filepath: str):
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.spots = data.get('spots', [])
    
    def validate_spot(self, spot: Dict[str, Any], game_state: Dict[str, Any]) -> Dict[str, Any]:
        result = {
            'spot_id': spot.get('id', 'unknown'),
            'passed': True,
            'errors': []
        }
        
        if 'expected_legal_actions' in spot:
            expected = set(spot['expected_legal_actions'])
            actual = set(game_state.get('legal_actions', []))
            
            if expected != actual:
                result['passed'] = False
                result['errors'].append({
                    'type': 'legal_actions_mismatch',
                    'expected': list(expected),
                    'actual': list(actual)
                })
        
        if 'expected_pot_size' in spot:
            expected_pot = spot['expected_pot_size']
            actual_pot = game_state.get('pot_size', 0)
            
            if abs(expected_pot - actual_pot) > 0.01:
                result['passed'] = False
                result['errors'].append({
                    'type': 'pot_size_mismatch',
                    'expected': expected_pot,
                    'actual': actual_pot
                })
        
        if 'expected_current_player' in spot:
            expected_player = spot['expected_current_player']
            actual_player = game_state.get('current_player', '')
            
            if expected_player != actual_player:
                result['passed'] = False
                result['errors'].append({
                    'type': 'current_player_mismatch',
                    'expected': expected_player,
                    'actual': actual_player
                })
        
        return result
    
    def validate_all_spots(self, game_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = []
        
        for idx, spot in enumerate(self.spots):
            if idx < len(game_states):
                result = self.validate_spot(spot, game_states[idx])
                results.append(result)
        
        total = len(results)
        passed = sum(1 for r in results if r['passed'])
        
        return {
            'total_spots': total,
            'passed': passed,
            'failed': total - passed,
            'pass_rate': passed / total if total > 0 else 0,
            'results': results
        }
    
    def create_spot_from_state(self, game_state: Dict[str, Any], 
                              spot_id: str, 
                              description: str = "") -> Dict[str, Any]:
        spot = {
            'id': spot_id,
            'description': description,
            'expected_legal_actions': game_state.get('legal_actions', []),
            'expected_current_player': game_state.get('current_player', ''),
            'pot_size': game_state.get('pot_size', 0)
        }
        
        return spot
    
    def save_spots(self, filepath: str, spots: List[Dict[str, Any]]):
        data = {
            'version': '1.0',
            'spots': spots
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def generate_sample_spots() -> List[Dict[str, Any]]:
        return [
            {
                'id': 'preflop_btn_vs_bb',
                'description': 'Button vs Big Blind preflop',
                'situation': 'preflop',
                'position': 'button',
                'expected_legal_actions': [0, 1, 2, 3, 4, 5],
                'notes': 'All actions should be available preflop'
            },
            {
                'id': 'flop_continuation_bet',
                'description': 'Flop after preflop raise',
                'situation': 'flop',
                'action_history': ['raise', 'call'],
                'expected_legal_actions': [0, 1, 2, 3, 4, 5],
                'notes': 'Standard continuation bet spot'
            },
            {
                'id': 'river_all_in_call',
                'description': 'River facing all-in',
                'situation': 'river',
                'action_history': ['bet', 'raise', 'all-in'],
                'expected_legal_actions': [0, 1],
                'notes': 'Only fold or call available facing all-in'
            },
            {
                'id': 'turn_check_back',
                'description': 'Turn after checking flop',
                'situation': 'turn',
                'action_history': ['check', 'check'],
                'expected_legal_actions': [0, 1, 2, 3, 4, 5],
                'notes': 'Full action range after checking'
            }
        ]
