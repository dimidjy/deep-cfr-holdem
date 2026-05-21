"""Shared engine factory — single source of truth for engine configuration.

Used by training, self-play, validation, and any other script that needs
a consistently configured FastEnhancedPokerEngine.
"""

from src.engine import FastEnhancedPokerEngine


def make_engine(num_players: int = 6, rake_pct: float = 0.0,
                rake_cap: float = 0.0, opponent_profile: bool = False,
                threat_range_equity: bool = False,
                equity_vs_random: bool = False):
    """Create a standard FastEnhancedPokerEngine with all features enabled.

    All scripts should use this instead of creating engines directly.
    """
    return FastEnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=True,
        behavioral_stats=True,
        opponent_profile=opponent_profile,
        threat_range_equity=threat_range_equity,
        equity_vs_random=equity_vs_random,
        rake_pct=rake_pct,
        rake_cap=rake_cap,
    )
