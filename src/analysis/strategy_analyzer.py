import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict


class StrategyAnalyzer:
    
    def __init__(self):
        self.action_frequencies = defaultdict(lambda: defaultdict(int))
        self.position_actions = defaultdict(lambda: defaultdict(int))
        self.situation_actions = defaultdict(lambda: defaultdict(int))
    
    def record_action(self, action: int, player: str, 
                     position: str = None, situation: str = None):
        
        self.action_frequencies[player][action] += 1
        
        if position:
            self.position_actions[position][action] += 1
        
        if situation:
            self.situation_actions[situation][action] += 1
    
    def get_action_distribution(self, player: str) -> Dict[int, float]:
        
        actions = self.action_frequencies[player]
        total = sum(actions.values())
        
        if total == 0:
            return {}
        
        return {action: count / total for action, count in actions.items()}
    
    def get_position_tendencies(self, position: str) -> Dict[int, float]:
        
        actions = self.position_actions[position]
        total = sum(actions.values())
        
        if total == 0:
            return {}
        
        return {action: count / total for action, count in actions.items()}
    
    def get_situation_tendencies(self, situation: str) -> Dict[int, float]:
        
        actions = self.situation_actions[situation]
        total = sum(actions.values())
        
        if total == 0:
            return {}
        
        return {action: count / total for action, count in actions.items()}
    
    def compute_aggression_factor(self, player: str) -> float:
        
        actions = self.action_frequencies[player]
        
        aggressive_actions = sum(actions.get(i, 0) for i in range(2, 10))
        passive_actions = actions.get(1, 0)
        
        if passive_actions == 0:
            return float('inf') if aggressive_actions > 0 else 0.0
        
        return aggressive_actions / passive_actions
    
    def compute_vpip(self, player: str) -> float:
        
        actions = self.action_frequencies[player]
        
        voluntary_actions = sum(actions.get(i, 0) for i in range(1, 10))
        total_actions = sum(actions.values())
        
        if total_actions == 0:
            return 0.0
        
        return voluntary_actions / total_actions
    
    def compute_pfr(self, player: str) -> float:
        
        actions = self.action_frequencies[player]
        
        raise_actions = sum(actions.get(i, 0) for i in range(2, 10))
        total_actions = sum(actions.values())
        
        if total_actions == 0:
            return 0.0
        
        return raise_actions / total_actions
    
    def get_summary(self, player: str) -> Dict[str, Any]:
        
        return {
            'action_distribution': self.get_action_distribution(player),
            'aggression_factor': self.compute_aggression_factor(player),
            'vpip': self.compute_vpip(player),
            'pfr': self.compute_pfr(player),
            'total_actions': sum(self.action_frequencies[player].values())
        }
    
    def compare_players(self, player1: str, player2: str) -> Dict[str, Any]:
        
        summary1 = self.get_summary(player1)
        summary2 = self.get_summary(player2)
        
        return {
            player1: summary1,
            player2: summary2,
            'differences': {
                'aggression_factor_diff': abs(summary1['aggression_factor'] - summary2['aggression_factor']),
                'vpip_diff': abs(summary1['vpip'] - summary2['vpip']),
                'pfr_diff': abs(summary1['pfr'] - summary2['pfr'])
            }
        }
    
    def reset(self):
        self.action_frequencies.clear()
        self.position_actions.clear()
        self.situation_actions.clear()
