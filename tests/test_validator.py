import pytest
import numpy as np
from src.engine import PokerValidator


class TestPokerValidator:
    
    def test_validate_action_legal(self):
        legal_actions = [0, 1, 2]
        assert PokerValidator.validate_action(1, legal_actions) is True
    
    def test_validate_action_illegal(self):
        legal_actions = [0, 1, 2]
        assert PokerValidator.validate_action(5, legal_actions) is False
    
    def test_validate_observation_valid(self):
        observation = {
            'observation': np.array([1, 2, 3]),
            'action_mask': np.array([1, 0, 1])
        }
        assert PokerValidator.validate_observation(observation) is True
    
    def test_validate_observation_missing_keys(self):
        observation = {
            'observation': np.array([1, 2, 3])
        }
        assert PokerValidator.validate_observation(observation) is False
    
    def test_validate_observation_wrong_types(self):
        observation = {
            'observation': [1, 2, 3],
            'action_mask': [1, 0, 1]
        }
        assert PokerValidator.validate_observation(observation) is False
    
    def test_validate_game_state_valid(self):
        state = {
            'observation': np.array([1, 2, 3]),
            'current_player': 'player_0',
            'legal_actions': [0, 1, 2]
        }
        assert PokerValidator.validate_game_state(state) is True
    
    def test_validate_game_state_terminal(self):
        state = {
            'observation': np.array([1, 2, 3]),
            'current_player': 'player_0',
            'legal_actions': [],
            'termination': True
        }
        assert PokerValidator.validate_game_state(state) is True
    
    def test_validate_pot_calculation(self):
        bets = [100, 200, 150]
        expected_pot = 450
        assert PokerValidator.validate_pot_calculation(bets, expected_pot) is True
    
    def test_validate_pot_calculation_mismatch(self):
        bets = [100, 200, 150]
        expected_pot = 500
        assert PokerValidator.validate_pot_calculation(bets, expected_pot) is False
    
    def test_validate_hand_ranking_pair(self):
        cards = [0, 13, 26, 39, 1]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "pair"
    
    def test_validate_hand_ranking_two_pair(self):
        cards = [0, 13, 1, 14, 2]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "two_pair"
    
    def test_validate_hand_ranking_three_of_kind(self):
        cards = [0, 13, 26, 1, 2]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "three_of_a_kind"
    
    def test_validate_hand_ranking_straight(self):
        cards = [0, 14, 28, 42, 4]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "straight"
    
    def test_validate_hand_ranking_flush(self):
        cards = [0, 1, 2, 3, 5]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "flush"
    
    def test_validate_hand_ranking_full_house(self):
        cards = [0, 13, 26, 1, 14]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking == "full_house"
    
    def test_validate_hand_ranking_four_of_kind(self):
        cards = [0, 13, 26, 39, 1]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking in ["four_of_a_kind", "pair"]
    
    def test_validate_hand_ranking_high_card(self):
        cards = [0, 14, 28, 42, 8]
        ranking = PokerValidator.validate_hand_ranking(cards)
        assert ranking in ["high_card", "straight"]
    
    def test_validate_chips_conservation(self):
        initial_chips = {'player_0': 1000, 'player_1': 1000}
        final_chips = {'player_0': 800, 'player_1': 900}
        pot = 300
        
        assert PokerValidator.validate_chips_conservation(initial_chips, final_chips, pot) is True
    
    def test_validate_chips_conservation_violation(self):
        initial_chips = {'player_0': 1000, 'player_1': 1000}
        final_chips = {'player_0': 800, 'player_1': 900}
        pot = 500
        
        assert PokerValidator.validate_chips_conservation(initial_chips, final_chips, pot) is False
