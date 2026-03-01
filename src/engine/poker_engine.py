import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
from typing import Dict, List, Tuple, Optional, Any


class PokerEngine:
    
    def __init__(self, num_players: int = 2):
        self.num_players = num_players
        self.env = None
        self.current_agent = None
        self.agents = []
        self.reset()
    
    def reset(self) -> Dict[str, Any]:
        self.env = texas_holdem_no_limit_v6.env(num_players=self.num_players)
        self.env.reset()
        self.agents = self.env.agents.copy()
        self.current_agent = self.env.agent_selection
        
        observation, reward, termination, truncation, info = self.env.last()
        
        return {
            'observation': observation,
            'legal_actions': self._get_legal_actions(),
            'current_player': self.current_agent,
            'info': info
        }
    
    def step(self, action: int) -> Dict[str, Any]:
        self.env.step(action)
        
        observation, reward, termination, truncation, info = self.env.last()
        self.current_agent = self.env.agent_selection
        
        done = termination or truncation
        
        return {
            'observation': observation,
            'reward': reward,
            'done': done,
            'termination': termination,
            'truncation': truncation,
            'legal_actions': self._get_legal_actions() if not done else [],
            'current_player': self.current_agent,
            'info': info
        }
    
    def _get_legal_actions(self) -> List[int]:
        if self.env.terminations[self.current_agent] or self.env.truncations[self.current_agent]:
            return []
        
        observation, _, _, _, _ = self.env.last()
        action_mask = observation.get('action_mask', None)
        
        if action_mask is not None:
            return [i for i, legal in enumerate(action_mask) if legal]
        
        return list(range(self.env.action_space(self.current_agent).n))
    
    def get_observation(self) -> np.ndarray:
        observation, _, _, _, _ = self.env.last()
        return observation['observation']
    
    def get_state(self) -> Dict[str, Any]:
        observation, reward, termination, truncation, info = self.env.last()
        
        return {
            'observation': observation,
            'reward': reward,
            'termination': termination,
            'truncation': truncation,
            'current_player': self.current_agent,
            'legal_actions': self._get_legal_actions(),
            'info': info,
            'agents': self.agents
        }
    
    def render(self, mode: str = 'human'):
        if hasattr(self.env, 'render'):
            self.env.render()
    
    def close(self):
        if self.env:
            self.env.close()
    
    def get_rewards(self) -> Dict[str, float]:
        return {agent: self.env.rewards[agent] for agent in self.agents}
    
    def is_done(self) -> bool:
        return all(self.env.terminations[agent] or self.env.truncations[agent] for agent in self.agents)
    
    def get_action_space_size(self) -> int:
        return self.env.action_space(self.agents[0]).n
    
    def get_observation_space_size(self) -> int:
        obs_space = self.env.observation_space(self.agents[0])
        return obs_space['observation'].shape[0]
    
    def _get_game(self):
        try:
            return self.env.unwrapped.env.game
        except:
            return None
    
    def _card_to_string(self, card) -> str:
        return f"{card.rank}{card.suit}"
    
    def get_player_cards(self, player: str) -> List[str]:
        if not self.env or player not in self.agents:
            return []
        
        try:
            game = self._get_game()
            if game:
                player_idx = self.agents.index(player)
                return [self._card_to_string(c) for c in game.players[player_idx].hand]
        except:
            pass
        
        return []
    
    def get_community_cards(self) -> List[str]:
        if not self.env:
            return []
        
        try:
            game = self._get_game()
            if game:
                return [self._card_to_string(c) for c in game.public_cards]
        except:
            pass
        
        return []
    
    def get_pot_size(self) -> int:
        if not self.env:
            return 0
        
        try:
            game = self._get_game()
            if game:
                return sum(p.in_chips for p in game.players)
        except:
            pass
        
        return 0
    
    def get_player_chips(self, player: str) -> int:
        if not self.env or player not in self.agents:
            return 0
        
        try:
            game = self._get_game()
            if game:
                player_idx = self.agents.index(player)
                return game.players[player_idx].remained_chips
        except:
            pass
        
        return 0
    
    def get_stage(self) -> str:
        try:
            game = self._get_game()
            if game:
                return game.stage.name.capitalize()
        except:
            pass
        
        return "Unknown"