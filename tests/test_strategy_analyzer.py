import pytest
import numpy as np
from src.analysis import StrategyAnalyzer


class TestStrategyAnalyzer:
    
    def test_analyzer_initialization(self):
        analyzer = StrategyAnalyzer()
        assert analyzer is not None
    
    def test_record_action(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(1, 'player_0')
        
        dist = analyzer.get_action_distribution('player_0')
        assert 1 in dist
        assert dist[1] == 1.0
    
    def test_multiple_actions(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(1, 'player_0')
        analyzer.record_action(2, 'player_0')
        analyzer.record_action(1, 'player_0')
        
        dist = analyzer.get_action_distribution('player_0')
        
        assert dist[1] == 2/3
        assert dist[2] == 1/3
    
    def test_position_tendencies(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(2, 'player_0', position='button')
        analyzer.record_action(2, 'player_0', position='button')
        analyzer.record_action(1, 'player_0', position='button')
        
        tendencies = analyzer.get_position_tendencies('button')
        
        assert tendencies[2] == 2/3
        assert tendencies[1] == 1/3
    
    def test_situation_tendencies(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(2, 'player_0', situation='preflop')
        analyzer.record_action(1, 'player_0', situation='preflop')
        
        tendencies = analyzer.get_situation_tendencies('preflop')
        
        assert tendencies[2] == 0.5
        assert tendencies[1] == 0.5
    
    def test_aggression_factor(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(1, 'player_0')
        analyzer.record_action(2, 'player_0')
        analyzer.record_action(2, 'player_0')
        
        af = analyzer.compute_aggression_factor('player_0')
        
        assert af == 2.0
    
    def test_vpip(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(0, 'player_0')
        analyzer.record_action(1, 'player_0')
        analyzer.record_action(2, 'player_0')
        
        vpip = analyzer.compute_vpip('player_0')
        
        assert vpip == 2/3
    
    def test_pfr(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(0, 'player_0')
        analyzer.record_action(1, 'player_0')
        analyzer.record_action(2, 'player_0')
        analyzer.record_action(3, 'player_0')
        
        pfr = analyzer.compute_pfr('player_0')
        
        assert pfr == 2/4
    
    def test_get_summary(self):
        analyzer = StrategyAnalyzer()
        
        for _ in range(10):
            analyzer.record_action(1, 'player_0')
        for _ in range(5):
            analyzer.record_action(2, 'player_0')
        
        summary = analyzer.get_summary('player_0')
        
        assert 'action_distribution' in summary
        assert 'aggression_factor' in summary
        assert 'vpip' in summary
        assert 'pfr' in summary
        assert 'total_actions' in summary
        assert summary['total_actions'] == 15
    
    def test_compare_players(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(1, 'player_0')
        analyzer.record_action(1, 'player_0')
        
        analyzer.record_action(2, 'player_1')
        analyzer.record_action(2, 'player_1')
        
        comparison = analyzer.compare_players('player_0', 'player_1')
        
        assert 'player_0' in comparison
        assert 'player_1' in comparison
        assert 'differences' in comparison
    
    def test_reset(self):
        analyzer = StrategyAnalyzer()
        
        analyzer.record_action(1, 'player_0')
        analyzer.reset()
        
        dist = analyzer.get_action_distribution('player_0')
        assert len(dist) == 0
