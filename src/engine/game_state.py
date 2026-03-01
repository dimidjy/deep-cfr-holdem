from typing import Dict, List, Any, Optional
import numpy as np


class GameState:
    
    def __init__(self, observation: np.ndarray, 
                 legal_actions: List[int],
                 current_player: str,
                 info: Dict[str, Any]):
        self.observation = observation
        self.legal_actions = legal_actions
        self.current_player = current_player
        self.info = info
        self.history = []
    
    def add_action(self, action: int, player: str):
        self.history.append({
            'action': action,
            'player': player
        })
    
    def get_history(self) -> List[Dict[str, Any]]:
        return self.history.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'observation': self.observation,
            'legal_actions': self.legal_actions,
            'current_player': self.current_player,
            'info': self.info,
            'history': self.history
        }


class HandHistory:
    
    def __init__(self):
        self.hands = []
        self.current_hand = None
    
    def start_hand(self):
        self.current_hand = {
            'actions': [],
            'states': [],
            'rewards': None
        }
    
    def add_action(self, action: int, player: str, state: Dict[str, Any]):
        if self.current_hand is None:
            self.start_hand()
        
        self.current_hand['actions'].append({
            'action': action,
            'player': player
        })
        self.current_hand['states'].append(state)
    
    def end_hand(self, rewards: Dict[str, float]):
        if self.current_hand is not None:
            self.current_hand['rewards'] = rewards
            self.hands.append(self.current_hand)
            self.current_hand = None
    
    def get_hands(self) -> List[Dict[str, Any]]:
        return self.hands.copy()
    
    def save_to_file(self, filepath: str):
        import json
        
        with open(filepath, 'w') as f:
            json.dump({'hands': self.hands}, f, indent=2, default=str)
    
    def load_from_file(self, filepath: str):
        import json
        
        with open(filepath, 'r') as f:
            data = json.load(f)
            self.hands = data.get('hands', [])
