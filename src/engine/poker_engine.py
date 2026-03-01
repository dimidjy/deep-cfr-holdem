import numpy as np
from pettingzoo.classic import texas_holdem_no_limit_v6
from typing import Dict, List, Tuple, Optional, Any


class PokerEngine:
    
    def __init__(self, num_players: int = 2, starting_chips: int = 10000):
        self.num_players = num_players
        self.starting_chips = starting_chips
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
