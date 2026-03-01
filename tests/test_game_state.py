import pytest
import numpy as np
from src.engine import GameState, HandHistory


class TestGameState:
    
    def test_game_state_initialization(self):
        observation = np.array([1, 2, 3])
        legal_actions = [0, 1, 2]
        current_player = 'player_0'
        info = {'test': 'data'}
        
        state = GameState(observation, legal_actions, current_player, info)
        
        assert np.array_equal(state.observation, observation)
        assert state.legal_actions == legal_actions
        assert state.current_player == current_player
        assert state.info == info
    
    def test_add_action(self):
        state = GameState(np.array([1, 2, 3]), [0, 1], 'player_0', {})
        
        state.add_action(1, 'player_0')
        
        history = state.get_history()
        assert len(history) == 1
        assert history[0]['action'] == 1
        assert history[0]['player'] == 'player_0'
    
    def test_multiple_actions(self):
        state = GameState(np.array([1, 2, 3]), [0, 1], 'player_0', {})
        
        state.add_action(1, 'player_0')
        state.add_action(2, 'player_1')
        state.add_action(0, 'player_0')
        
        history = state.get_history()
        assert len(history) == 3
    
    def test_to_dict(self):
        state = GameState(np.array([1, 2, 3]), [0, 1], 'player_0', {'key': 'value'})
        state.add_action(1, 'player_0')
        
        state_dict = state.to_dict()
        
        assert 'observation' in state_dict
        assert 'legal_actions' in state_dict
        assert 'current_player' in state_dict
        assert 'info' in state_dict
        assert 'history' in state_dict


class TestHandHistory:
    
    def test_hand_history_initialization(self):
        history = HandHistory()
        assert history.hands == []
        assert history.current_hand is None
    
    def test_start_hand(self):
        history = HandHistory()
        history.start_hand()
        
        assert history.current_hand is not None
        assert 'actions' in history.current_hand
        assert 'states' in history.current_hand
        assert 'rewards' in history.current_hand
    
    def test_add_action(self):
        history = HandHistory()
        history.start_hand()
        
        state = {'test': 'state'}
        history.add_action(1, 'player_0', state)
        
        assert len(history.current_hand['actions']) == 1
        assert len(history.current_hand['states']) == 1
    
    def test_end_hand(self):
        history = HandHistory()
        history.start_hand()
        history.add_action(1, 'player_0', {})
        
        rewards = {'player_0': 100, 'player_1': -100}
        history.end_hand(rewards)
        
        assert len(history.hands) == 1
        assert history.current_hand is None
        assert history.hands[0]['rewards'] == rewards
    
    def test_multiple_hands(self):
        history = HandHistory()
        
        for i in range(3):
            history.start_hand()
            history.add_action(i, f'player_{i % 2}', {})
            history.end_hand({'player_0': 0, 'player_1': 0})
        
        assert len(history.hands) == 3
    
    def test_get_hands(self):
        history = HandHistory()
        history.start_hand()
        history.add_action(1, 'player_0', {})
        history.end_hand({'player_0': 100, 'player_1': -100})
        
        hands = history.get_hands()
        
        assert len(hands) == 1
        assert isinstance(hands, list)
