"""
Comprehensive tests for FastPokerEngine and FastEnhancedPokerEngine.

Tests:
1. Basic game flow (reset, step, done)
2. Legal actions validity
3. Rewards sum to zero (zero-sum game)
4. Feature vector dimensions and ranges
5. Deterministic replay via save/restore
6. Many random games complete without errors
7. Specific scenarios (fold, all-in, showdown)
8. Performance comparison with PettingZoo
9. Comparison of feature dimensions between engines
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import (
    FastPokerEngine,
    FastEnhancedPokerEngine,
)


# ── 1. Basic game flow ──────────────────────────────────────────────────────

class TestBasicFlow:
    def test_create_engine(self):
        eng = FastPokerEngine(num_players=6)
        assert eng.num_players == 6
        assert len(eng.agents) == 6
        assert not eng.is_done()

    def test_reset_returns_dict(self):
        eng = FastPokerEngine(num_players=6)
        result = eng.reset()
        assert 'legal_actions' in result
        assert 'current_player' in result

    def test_step_and_finish(self):
        eng = FastPokerEngine(num_players=2)
        eng.reset()
        # Everyone folds until done
        steps = 0
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            eng.step(legal[0])  # fold if available
            steps += 1
            if steps > 100:
                pytest.fail("Game didn't end in 100 steps")

    def test_close_is_safe(self):
        eng = FastPokerEngine(num_players=6)
        eng.close()  # should not raise

    def test_action_space_size(self):
        eng = FastPokerEngine(num_players=6)
        # fold, call, raise_half, raise_pot — all-in enters only via
        # runtime promotions, never as a network head.
        assert eng.get_action_space_size() == 4

    def test_agents_list(self):
        eng = FastPokerEngine(num_players=6)
        assert eng.agents == [f'player_{i}' for i in range(6)]

    def test_two_players(self):
        eng = FastPokerEngine(num_players=2)
        eng.reset()
        assert not eng.is_done()
        legal = eng.get_legal_actions()
        assert len(legal) > 0


# ── 2. Legal actions validity ────────────────────────────────────────────────

class TestLegalActions:
    def test_legal_actions_are_in_range(self):
        eng = FastPokerEngine(num_players=6)
        for _ in range(100):
            eng.reset()
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                for a in legal:
                    assert 0 <= a <= 4, f"Illegal action {a}"
                eng.step(np.random.choice(legal))

    def test_fold_and_checkcall_always_present(self):
        eng = FastPokerEngine(num_players=6)
        for _ in range(50):
            eng.reset()
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                # At least fold or check/call should be present
                assert 0 in legal or 1 in legal, f"Neither fold nor call in {legal}"
                eng.step(np.random.choice(legal))

    def test_no_duplicate_actions(self):
        eng = FastPokerEngine(num_players=6)
        for _ in range(50):
            eng.reset()
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                assert len(legal) == len(set(legal)), f"Duplicates in {legal}"
                eng.step(np.random.choice(legal))


# ── 3. Rewards sum to zero ──────────────────────────────────────────────────

class TestRewards:
    def test_rewards_sum_to_zero(self):
        eng = FastPokerEngine(num_players=6)
        for _ in range(200):
            eng.reset()
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))
            rewards = eng.get_rewards()
            total = sum(rewards.values())
            assert abs(total) < 0.01, f"Rewards don't sum to zero: {rewards} (sum={total})"

    def test_rewards_sum_to_zero_2p(self):
        eng = FastPokerEngine(num_players=2)
        for _ in range(200):
            eng.reset()
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))
            rewards = eng.get_rewards()
            total = sum(rewards.values())
            assert abs(total) < 0.01, f"Rewards don't sum to zero: {rewards}"

    def test_fold_reward_reasonable(self):
        """When folding preflop, loss should be small (0 or blind amount)."""
        eng = FastPokerEngine(num_players=6)
        fold_rewards = []
        for _ in range(100):
            eng.reset()
            # Current player folds
            folder = eng.current_agent
            legal = eng.get_legal_actions()
            if 0 in legal:
                eng.step(0)
                # Play rest randomly
                while not eng.is_done():
                    legal = eng.get_legal_actions()
                    if not legal:
                        break
                    eng.step(np.random.choice(legal))
                rewards = eng.get_rewards()
                # The folder should lose 0 or at most their blind
                fold_rewards.append(rewards[folder])

        # Folds should result in 0 or small negative reward (blind at most)
        for r in fold_rewards:
            assert r <= 0.1, f"Folder gained chips? reward={r}"
            assert r >= -3, f"Folder lost too much? reward={r}"


# ── 4. Feature vectors ──────────────────────────────────────────────────────

class TestFeatures:
    def test_enhanced_obs_dimensions(self):
        eng = FastEnhancedPokerEngine(num_players=6)
        obs_size = eng.get_observation_space_size()
        eng.reset()
        while not eng.is_done():
            obs = eng.get_observation()
            assert obs.shape == (obs_size,), f"Wrong shape: {obs.shape} vs ({obs_size},)"
            legal = eng.get_legal_actions()
            if not legal:
                break
            eng.step(np.random.choice(legal))

    def test_feature_values_in_range(self):
        eng = FastEnhancedPokerEngine(num_players=6)
        for _ in range(50):
            eng.reset()
            while not eng.is_done():
                obs = eng.get_observation()
                assert np.all(np.isfinite(obs)), f"Non-finite values in obs: {obs}"
                assert np.all(obs >= -0.1), f"Values below -0.1: {obs.min()}"
                assert np.all(obs <= 1.5), f"Values above 1.5: {obs.max()}"
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))


# ── 5. Deterministic replay ─────────────────────────────────────────────────

class TestReplay:
    def test_save_restore_produces_same_game(self):
        """Save initial state, play a game, restore, replay — same results."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()

        # Save state
        game = eng._get_game()
        saved_state = game.np_random.get_state()
        saved_dealer = game.dealer_id

        # Play a game, record actions and observations
        actions_taken = []
        legal_seen = []
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            legal_seen.append(list(legal))
            a = np.random.choice(legal)
            actions_taken.append(a)
            eng.step(a)
        rewards1 = eng.get_rewards()

        # Restore and replay
        eng2 = FastPokerEngine(num_players=6)
        game2 = eng2._get_game()
        game2.np_random.set_state(saved_state)
        game2.dealer_id = saved_dealer
        eng2.reset()

        for i, a in enumerate(actions_taken):
            legal = eng2.get_legal_actions()
            assert legal == legal_seen[i], \
                f"Step {i}: legal actions differ: {legal} vs {legal_seen[i]}"
            eng2.step(a)

        rewards2 = eng2.get_rewards()
        for agent in eng.agents:
            assert abs(rewards1[agent] - rewards2[agent]) < 0.01, \
                f"Reward mismatch for {agent}: {rewards1[agent]} vs {rewards2[agent]}"

    def test_replay_many_times(self):
        """Replay same game 5 times — always same result."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        saved = eng._get_game().np_random.get_state()

        # Play once
        actions = []
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            a = legal[0]
            actions.append(a)
            eng.step(a)
        expected = eng.get_rewards()

        # Replay 5 times
        for trial in range(5):
            eng2 = FastPokerEngine(num_players=6)
            eng2._get_game().np_random.set_state(saved)
            eng2.reset()
            for a in actions:
                eng2.step(a)
            actual = eng2.get_rewards()
            for agent in eng.agents:
                assert abs(expected[agent] - actual[agent]) < 0.01, \
                    f"Trial {trial}: reward mismatch for {agent}"


# ── 6. Stress test (many random games) ──────────────────────────────────────

class TestStress:
    def test_1000_random_games_6p(self):
        eng = FastPokerEngine(num_players=6)
        for game_i in range(1000):
            eng.reset()
            steps = 0
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))
                steps += 1
                if steps > 200:
                    pytest.fail(f"Game {game_i} didn't end in 200 steps")
            rewards = eng.get_rewards()
            total = sum(rewards.values())
            assert abs(total) < 0.01, f"Game {game_i}: rewards sum={total}"

    def test_1000_random_games_2p(self):
        eng = FastPokerEngine(num_players=2)
        for game_i in range(1000):
            eng.reset()
            steps = 0
            while not eng.is_done():
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))
                steps += 1
                if steps > 200:
                    pytest.fail(f"Game {game_i} didn't end in 200 steps")
            rewards = eng.get_rewards()
            total = sum(rewards.values())
            assert abs(total) < 0.01, f"Game {game_i}: rewards sum={total}"

    def test_enhanced_1000_games(self):
        eng = FastEnhancedPokerEngine(num_players=6)
        obs_size = eng.get_observation_space_size()
        for _ in range(1000):
            eng.reset()
            while not eng.is_done():
                obs = eng.get_observation()
                assert obs.shape == (obs_size,)
                assert np.all(np.isfinite(obs))
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))


# ── 7. Specific scenarios ───────────────────────────────────────────────────

class TestScenarios:
    def test_all_fold(self):
        """All but one player folds — game ends, one winner."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            if 0 in legal:
                eng.step(0)  # fold
            else:
                eng.step(legal[0])
        assert eng.is_done()
        rewards = eng.get_rewards()
        assert abs(sum(rewards.values())) < 0.01

    def test_all_call_to_showdown(self):
        """All players check/call — goes to showdown."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            if 1 in legal:
                eng.step(1)  # check/call
            else:
                eng.step(legal[0])
        assert eng.is_done()
        rewards = eng.get_rewards()
        assert abs(sum(rewards.values())) < 0.01

    def test_allin_action(self):
        """All-in action works correctly."""
        eng = FastPokerEngine(num_players=2)
        for _ in range(50):
            eng.reset()
            legal = eng.get_legal_actions()
            if 4 in legal:
                eng.step(4)  # all-in
                # Other player acts
                while not eng.is_done():
                    legal2 = eng.get_legal_actions()
                    if not legal2:
                        break
                    eng.step(np.random.choice(legal2))
                assert eng.is_done()
                rewards = eng.get_rewards()
                assert abs(sum(rewards.values())) < 0.01



# ── 9. _get_game proxy ──────────────────────────────────────────────────────

class TestGameProxy:
    def test_get_game_returns_proxy(self):
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        game = eng._get_game()
        assert hasattr(game, 'np_random')
        assert hasattr(game, 'dealer_id')
        assert hasattr(game, 'big_blind')
        assert hasattr(game, 'players')
        assert hasattr(game, 'public_cards')
        assert hasattr(game, 'stage')
        assert hasattr(game, 'round')

    def test_proxy_players(self):
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        game = eng._get_game()
        players = game.players
        assert len(players) == 6
        for p in players:
            assert hasattr(p, 'hand')
            assert hasattr(p, 'remained_chips')
            assert hasattr(p, 'in_chips')
            assert hasattr(p, 'status')
            assert len(p.hand) == 2
            for card in p.hand:
                assert hasattr(card, 'rank')
                assert hasattr(card, 'suit')
                assert card.rank in '23456789TJQKA'
                assert card.suit in 'SHDC'

    def test_proxy_round(self):
        eng = FastPokerEngine(num_players=6)
        eng.reset()
        game = eng._get_game()
        r = game.round
        assert hasattr(r, 'raised')
        assert hasattr(r, 'not_raise_num')
        assert len(r.raised) == 6


# ── 10. Deep feature parity comparison ─────────────────────────────────────

class TestFeatureParity:
    """Compare feature distributions between old and new engines."""

    def _collect_features(self, engine_class, n_games=200, num_players=6):
        """Play n_games, collect all observation vectors."""
        eng = engine_class(num_players=num_players)
        all_obs = []
        for _ in range(n_games):
            eng.reset()
            while not eng.is_done():
                obs = eng.get_observation()
                all_obs.append(obs.copy())
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))
        if hasattr(eng, 'close'):
            eng.close()
        return np.array(all_obs)

    def test_position_features_one_hot(self):
        """Position features should be valid one-hot + normalized values."""
        eng = FastEnhancedPokerEngine(num_players=6)
        for _ in range(100):
            eng.reset()
            while not eng.is_done():
                obs = eng.get_observation()
                # First 6 dims: one-hot position
                pos_onehot = obs[:6]
                assert abs(pos_onehot.sum() - 1.0) < 1e-5, f"One-hot sum={pos_onehot.sum()}"
                assert pos_onehot.min() >= 0
                assert pos_onehot.max() <= 1.0
                # Dim 6: normalized position [0, 1)
                assert 0 <= obs[6] <= 1.0
                # Dim 7: is_button (0 or 1)
                assert obs[7] in (0.0, 1.0)
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))

    def test_street_features_one_hot(self):
        """Street features should always be a valid one-hot vector."""
        eng = FastEnhancedPokerEngine(num_players=6)
        from src.obs_schema import index_of
        street_start = index_of('is_preflop')
        for _ in range(100):
            eng.reset()
            while not eng.is_done():
                obs = eng.get_observation()
                street = obs[street_start:street_start + 4]
                assert abs(street.sum() - 1.0) < 1e-5, f"Street one-hot sum={street.sum()}: {street}"
                assert street.min() >= 0
                assert street.max() <= 1.0
                legal = eng.get_legal_actions()
                if not legal:
                    break
                eng.step(np.random.choice(legal))

    def test_hand_strength_features_consistency(self):
        """Hand strength features should follow poker logic."""
        eng = FastEnhancedPokerEngine(num_players=6)
        # Feature layout: pos(8) + pot_odds(3) + hand_str(5)
        hs_start = 8 + 3  # = 11
        for _ in range(100):
            eng.reset()
            obs = eng.get_observation()
            hs = obs[hs_start:hs_start + 5]
            # is_pair: 0 or 1
            assert hs[0] in (0.0, 1.0), f"is_pair={hs[0]}"
            # is_suited: 0 or 1
            assert hs[1] in (0.0, 1.0), f"is_suited={hs[1]}"
            # high_card_norm: [0, 1]
            assert 0 <= hs[2] <= 1.0, f"high_card={hs[2]}"
            # low_card_norm: [0, 1], <= high_card
            assert 0 <= hs[3] <= 1.0
            assert hs[3] <= hs[2] + 1e-5, f"low > high: {hs[3]} > {hs[2]}"
            # rank_gap: [0, 1], for pairs should be 0
            if hs[0] == 1.0:
                assert abs(hs[4]) < 1e-5, f"Pair but gap={hs[4]}"

    def test_call_amount_matches_old_engine_preflop(self):
        """At preflop, call amount should match between engines."""
        # Can't test exact same hand, but can verify logic with known states
        # On preflop with no raise: call_amount should be BB (2) for non-blind positions
        eng = FastEnhancedPokerEngine(num_players=6)
        for _ in range(50):
            eng.reset()
            # Check pot_odds feature (indices 8,9,10) - specifically pot_odds at index 10
            obs = eng.get_observation()
            pot_odds = obs[10]
            # If there's a call to make, pot_odds > 0
            ps = eng._cur_ps()
            max_bet = max(p.bet_chips for p in eng._state.players_state)
            my_in = ps.pot_chips + ps.bet_chips
            call_amt = max(0, max_bet - my_in)
            if call_amt > 0:
                expected_po = eng._state.pot / (eng._state.pot + call_amt)
                assert abs(pot_odds - expected_po) < 0.01, \
                    f"pot_odds mismatch: obs={pot_odds}, expected={expected_po}"


# ── 11. MCCFR replay pattern test ──────────────────────────────────────────

class TestMCCFRReplay:
    """Test the exact save/restore pattern used in train_gpu.py."""

    def test_branch_replay_gives_same_cards(self):
        """Simulate MCCFR branching: save state, play action A, restore, play action B.
        Both branches should have identical cards dealt."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()

        # Save (mimics Phase 1 of train_gpu.py)
        game = eng._get_game()
        saved_state = game.np_random.get_state()

        # Get initial legal actions and cards
        cards_before = eng.get_player_cards(eng.current_agent)
        community_before = eng.get_community_cards()

        # Branch A: fold
        eng_a = FastPokerEngine(num_players=6)
        eng_a._get_game().np_random.set_state(saved_state)
        eng_a.reset()
        cards_a = eng_a.get_player_cards(eng_a.current_agent)
        community_a = eng_a.get_community_cards()

        # Branch B: call
        eng_b = FastPokerEngine(num_players=6)
        eng_b._get_game().np_random.set_state(saved_state)
        eng_b.reset()
        cards_b = eng_b.get_player_cards(eng_b.current_agent)
        community_b = eng_b.get_community_cards()

        assert cards_a == cards_before, f"Branch A cards differ: {cards_a} vs {cards_before}"
        assert cards_b == cards_before, f"Branch B cards differ: {cards_b} vs {cards_before}"
        assert community_a == community_before
        assert community_b == community_before

    def test_branch_with_history_replay(self):
        """Replay history + branch action, as in train_gpu.py _fresh_engine_at."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()

        game = eng._get_game()
        saved = game.np_random.get_state()

        # Play a few actions, record history
        history = []
        for _ in range(3):
            if eng.is_done():
                break
            legal = eng.get_legal_actions()
            if not legal:
                break
            a = legal[0]
            history.append(a)
            eng.step(a)

        if eng.is_done():
            return  # too short, skip

        # Get state at this point
        expected_legal = eng.get_legal_actions()
        expected_agent = eng.current_agent
        expected_cards = eng.get_player_cards(expected_agent)

        # Replay from saved state + history
        eng2 = FastPokerEngine(num_players=6)
        eng2._get_game().np_random.set_state(saved)
        eng2.reset()
        for a in history:
            eng2.step(a)

        assert eng2.current_agent == expected_agent
        assert eng2.get_legal_actions() == expected_legal
        assert eng2.get_player_cards(expected_agent) == expected_cards

    def test_multiple_branches_from_same_state(self):
        """Create 10 branches from same saved state — all identical initial state."""
        eng = FastPokerEngine(num_players=6)
        eng.reset()

        saved = eng._get_game().np_random.get_state()
        reference_cards = {a: eng.get_player_cards(a) for a in eng.agents}
        reference_community = eng.get_community_cards()

        for _ in range(10):
            eng_b = FastPokerEngine(num_players=6)
            eng_b._get_game().np_random.set_state(saved)
            eng_b.reset()
            for a in eng.agents:
                assert eng_b.get_player_cards(a) == reference_cards[a]
            assert eng_b.get_community_cards() == reference_community


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
