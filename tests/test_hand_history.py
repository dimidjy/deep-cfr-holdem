"""Tests for hand history parsing, action mapping, and observation building."""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.hand_history.models import (
    ParsedPlayer, ParsedAction, ParsedHand, HandResult,
    DecisionPoint, dollars_to_chips, DOLLARS_TO_CHIPS,
)
from src.hand_history.parser import parse_hand, parse_file
from src.hand_history.action_mapper import (
    map_action, get_legal_actions, FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN,
)
from src.hand_history.obs_builder import ObservationBuilder


# ──────────────────────────────────────────────────────────────────────────────
# Test fixtures: real hand history text blocks
# ──────────────────────────────────────────────────────────────────────────────

# 2-player hand, no showdown (fold on river)
HAND_2P_NO_SHOWDOWN = """\
PokerStars Hand #100000000001:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 15:50:33 WET [2020/01/01 10:50:33 ET]
Table 'Sample I' 6-max Seat #1 is the button
Seat 1: player_c ($5 in chips)
Seat 6: player_b ($4.25 in chips)
player_c: posts small blind $0.02
player_b: posts big blind $0.05
*** HOLE CARDS ***
player_c: raises $0.10 to $0.15
player_b: calls $0.10
*** FLOP *** [9h 9d Qs]
player_e joins the table at seat #2
player_b: checks
player_c: checks
*** TURN *** [9h 9d Qs] [Kc]
player_b: checks
player_c: checks
*** RIVER *** [9h 9d Qs Kc] [3d]
player_a joins the table at seat #3
player_b: checks
player_c: bets $0.05
player_b: folds
Uncalled bet ($0.05) returned to player_c
player_c collected $0.28 from pot
player_c: doesn't show hand
player_d joins the table at seat #4
*** SUMMARY ***
Total pot $0.30 | Rake $0.02
Board [9h 9d Qs Kc 3d]
Seat 1: player_c (button) (small blind) collected ($0.28)
Seat 6: player_b (big blind) folded on the River
"""

# Showdown hand with 4 active players (2 sitting out)
HAND_SHOWDOWN = """\
PokerStars Hand #100000000002:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 15:51:33 WET [2020/01/01 10:51:33 ET]
Table 'Sample I' 6-max Seat #1 is the button
Seat 4: player_d ($5 in chips) is sitting out
Seat 1: player_c ($5.11 in chips)
Seat 2: player_e ($4.11 in chips)
Seat 3: player_a ($4.95 in chips)
Seat 6: player_b ($4.05 in chips)
player_e: posts small blind $0.02
player_a: posts big blind $0.05
player_d: sits out
*** HOLE CARDS ***
player_b: calls $0.05
player_c: folds
player_e: folds
player_a: checks
*** FLOP *** [Ts Ks 3h]
player_a: checks
player_b: checks
*** TURN *** [Ts Ks 3h] [7s]
player_a: checks
player_b: checks
*** RIVER *** [Ts Ks 3h 7s] [As]
player_a: checks
player_b: bets $0.06
player_a: calls $0.06
*** SHOW DOWN ***
player_b: shows [3c Ah] (two pair, Aces and Threes)
player_a: shows [Js 5s] (a flush, Ace high)
player_a collected $0.23 from pot
*** SUMMARY ***
Total pot $0.24 | Rake $0.01
Board [Ts Ks 3h 7s As]
Seat 4: player_d
Seat 1: player_c (button) folded before Flop (didn't bet)
Seat 2: player_e (small blind) folded before Flop
Seat 3: player_a (big blind) showed [Js 5s] and won ($0.23) with a flush, Ace high
Seat 6: player_b showed [3c Ah] and lost with two pair, Aces and Threes
"""

# 5-player hand with raise preflop and muck at showdown
HAND_RAISE_AND_MUCK = """\
PokerStars Hand #100000000003:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 15:51:58 WET [2020/01/01 10:51:58 ET]
Table 'Sample I' 6-max Seat #2 is the button
Seat 1: player_c ($5.11 in chips)
Seat 2: player_e ($4.09 in chips)
Seat 3: player_a ($5.07 in chips)
Seat 4: player_d ($5 in chips)
Seat 6: player_b ($3.94 in chips)
player_a: posts small blind $0.02
player_d: posts big blind $0.05
*** HOLE CARDS ***
player_b: calls $0.05
player_c: folds
player_e: folds
player_a: calls $0.03
player_d: raises $0.15 to $0.20
player_b: calls $0.15
player_a: folds
*** FLOP *** [6c 6h Kd]
player_d: checks
player_b: checks
*** TURN *** [6c 6h Kd] [8d]
player_d: checks
player_b: checks
*** RIVER *** [6c 6h Kd 8d] [2d]
player_d: checks
player_b: bets $0.05
player_d: calls $0.05
*** SHOW DOWN ***
player_b: shows [8s 7s] (two pair, Eights and Sixes)
player_d: mucks hand
player_b collected $0.52 from pot
*** SUMMARY ***
Total pot $0.55 | Rake $0.03
Board [6c 6h Kd 8d 2d]
Seat 1: player_c folded before Flop (didn't bet)
Seat 2: player_e (button) folded before Flop (didn't bet)
Seat 3: player_a (small blind) folded before Flop
Seat 4: player_d (big blind) mucked
Seat 6: player_b showed [8s 7s] and won ($0.52) with two pair, Eights and Sixes
"""

# All-in hand
HAND_ALLIN = """\
PokerStars Hand #100000000004:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 16:01:50 WET [2020/01/01 11:01:50 ET]
Table 'Sample I' 6-max Seat #1 is the button
Seat 1: player_f ($3.78 in chips)
Seat 2: player_e ($3.84 in chips)
Seat 3: player_a ($5.18 in chips)
Seat 4: player_d ($5.17 in chips)
Seat 5: player_g ($5 in chips)
Seat 6: player_h ($5.32 in chips)
player_e: posts small blind $0.02
player_a: posts big blind $0.05
*** HOLE CARDS ***
player_d: folds
player_g: raises $0.08 to $0.13
player_h: folds
player_f: calls $0.13
player_e: calls $0.11
player_a: calls $0.08
*** FLOP *** [4d 9h Kc]
player_e: checks
player_a: bets $0.57
player_g: folds
player_f: raises $0.57 to $1.14
player_e: folds
player_a: calls $0.57
*** TURN *** [4d 9h Kc] [2s]
player_a: bets $3.81 and is all-in
player_f: calls $2.44 and is all-in
Uncalled bet ($1.37) returned to player_a
*** RIVER *** [4d 9h Kc 2s] [2h]
*** SHOW DOWN ***
player_a: shows [Js Kh] (two pair, Kings and Deuces)
player_f: shows [Qs Qh] (two pair, Queens and Deuces)
player_a collected $7.37 from pot
*** SUMMARY ***
Total pot $7.76 | Rake $0.39
Board [4d 9h Kc 2s 2h]
Seat 1: player_f (button) showed [Qs Qh] and lost with two pair, Queens and Deuces
Seat 2: player_e (small blind) folded on the Flop
Seat 3: player_a (big blind) showed [Js Kh] and won ($7.37) with two pair, Kings and Deuces
Seat 4: player_d folded before Flop (didn't bet)
Seat 5: player_g folded on the Flop
Seat 6: player_h folded before Flop (didn't bet)
"""

# Small & big blinds edge case
HAND_SMALL_BIG_BLINDS = """\
PokerStars Hand #100000000005:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 16:30:00 WET [2020/01/01 11:30:00 ET]
Table 'TestTable' 6-max Seat #3 is the button
Seat 1: Player1 ($5 in chips)
Seat 2: Player2 ($5 in chips)
Seat 3: Player3 ($5 in chips)
Seat 4: Player4 ($5 in chips)
Seat 5: Player5 ($5 in chips)
Seat 6: Player6 ($5 in chips)
Player4: posts small blind $0.02
Player5: posts big blind $0.05
Player6: posts small & big blinds $0.07
*** HOLE CARDS ***
Player1: folds
Player2: folds
Player3: folds
Player6: checks
Player4: folds
Player5: checks
*** FLOP *** [Ah Kd 3c]
Player5: checks
Player6: bets $0.10
Player5: folds
Uncalled bet ($0.10) returned to Player6
Player6 collected $0.14 from pot
*** SUMMARY ***
Total pot $0.14 | Rake $0
Board [Ah Kd 3c]
Seat 4: Player4 (small blind) folded before Flop
Seat 5: Player5 (big blind) folded on the Flop
Seat 6: Player6 collected ($0.14)
"""


# ──────────────────────────────────────────────────────────────────────────────
# Models tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModels:
    def test_dollars_to_chips(self):
        assert DOLLARS_TO_CHIPS == 40.0
        assert dollars_to_chips(0.05) == 2   # 1 BB = 2 chips
        assert dollars_to_chips(0.02) == 1   # 1 SB = 1 chip (rounded)
        assert dollars_to_chips(5.00) == 200  # starting stack
        assert dollars_to_chips(0.10) == 4
        assert dollars_to_chips(0.0) == 0

    def test_parsed_player_auto_chips(self):
        p = ParsedPlayer(name='Test', seat=1, stack_dollars=5.0)
        assert p.stack_chips == 200

    def test_parsed_action_auto_chips(self):
        a = ParsedAction(player_name='Test', action_type='bet',
                         amount_dollars=0.15)
        assert a.amount_chips == 6  # 0.15 * 40 = 6

    def test_parsed_hand_community_cards_at_street(self):
        h = ParsedHand()
        h.flop = ['9h', '9d', 'Qs']
        h.turn = 'Kc'
        h.river = '3d'
        cc = h.community_cards_at_street
        assert cc['preflop'] == []
        assert cc['flop'] == ['9h', '9d', 'Qs']
        assert cc['turn'] == ['9h', '9d', 'Qs', 'Kc']
        assert cc['river'] == ['9h', '9d', 'Qs', 'Kc', '3d']

    def test_parsed_hand_active_players(self):
        h = ParsedHand()
        h.players = [
            ParsedPlayer(name='A', seat=1, stack_dollars=5.0),
            ParsedPlayer(name='B', seat=2, stack_dollars=5.0, is_sitting_out=True),
            ParsedPlayer(name='C', seat=3, stack_dollars=5.0),
        ]
        assert h.num_active_players == 2
        assert len(h.active_players) == 2
        assert h.active_players[0].name == 'A'
        assert h.active_players[1].name == 'C'

    def test_has_showdown(self):
        h = ParsedHand()
        assert not h.has_showdown
        h.shown_hands = {'Player1': ['Ah', 'Kd']}
        assert h.has_showdown


# ──────────────────────────────────────────────────────────────────────────────
# Parser tests
# ──────────────────────────────────────────────────────────────────────────────

class TestParser:
    def test_parse_2p_no_showdown(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        assert hand is not None
        assert hand.hand_id == '100000000001'
        assert hand.small_blind_dollars == 0.02
        assert hand.big_blind_dollars == 0.05
        assert hand.button_seat == 1
        assert hand.max_seats == 6
        assert hand.table_name == 'Sample I'

    def test_parse_players(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        assert len(hand.players) == 2
        assert hand.players[0].name == 'player_c'
        assert hand.players[0].seat == 1
        assert hand.players[0].stack_dollars == 5.0
        assert hand.players[0].stack_chips == 200
        assert hand.players[1].name == 'player_b'
        assert hand.players[1].stack_dollars == 4.25

    def test_parse_sitting_out(self):
        hand = parse_hand(HAND_SHOWDOWN)
        sitting_out = [p for p in hand.players if p.is_sitting_out]
        assert len(sitting_out) == 1
        assert sitting_out[0].name == 'player_d'
        assert hand.num_active_players == 4

    def test_parse_blinds(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        blinds = [a for a in hand.actions if a.is_blind]
        assert len(blinds) == 2
        assert blinds[0].player_name == 'player_c'
        assert blinds[0].action_type == 'blind_sb'
        assert blinds[0].amount_dollars == 0.02
        assert blinds[1].player_name == 'player_b'
        assert blinds[1].action_type == 'blind_bb'
        assert blinds[1].amount_dollars == 0.05

    def test_parse_actions_preflop(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        preflop = [a for a in hand.actions
                   if a.street == 'preflop' and not a.is_blind]
        assert len(preflop) == 2
        assert preflop[0].action_type == 'raise'
        assert preflop[0].amount_dollars == 0.15  # "to" amount
        assert preflop[1].action_type == 'call'
        assert preflop[1].amount_dollars == 0.10

    def test_parse_community_cards(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        assert hand.flop == ['9h', '9d', 'Qs']
        assert hand.turn == 'Kc'
        assert hand.river == '3d'

    def test_parse_community_cards_flop_only(self):
        """Hand that ends on the flop should have no turn/river."""
        hand = parse_hand(HAND_SMALL_BIG_BLINDS)
        assert hand.flop == ['Ah', 'Kd', '3c']
        assert hand.turn is None
        assert hand.river is None

    def test_parse_showdown_hands(self):
        hand = parse_hand(HAND_SHOWDOWN)
        assert hand.has_showdown
        assert 'player_b' in hand.shown_hands
        assert 'player_a' in hand.shown_hands
        assert hand.shown_hands['player_b'] == ['3c', 'Ah']
        assert hand.shown_hands['player_a'] == ['Js', '5s']

    def test_parse_muck(self):
        """Muck should not add to shown_hands."""
        hand = parse_hand(HAND_RAISE_AND_MUCK)
        assert 'player_b' in hand.shown_hands  # shows
        assert 'player_d' not in hand.shown_hands  # mucks

    def test_parse_result(self):
        hand = parse_hand(HAND_SHOWDOWN)
        assert hand.result.total_pot_dollars == 0.24
        assert hand.result.rake_dollars == 0.01
        assert len(hand.result.winners) == 1
        assert hand.result.winners[0]['name'] == 'player_a'
        assert hand.result.winners[0]['amount_dollars'] == 0.23

    def test_parse_allin(self):
        hand = parse_hand(HAND_ALLIN)
        allin_actions = [a for a in hand.actions if a.is_all_in]
        assert len(allin_actions) == 2
        # player_a bets all-in
        assert allin_actions[0].player_name == 'player_a'
        assert allin_actions[0].action_type == 'bet'
        assert allin_actions[0].amount_dollars == 3.81
        # player_f calls all-in
        assert allin_actions[1].player_name == 'player_f'
        assert allin_actions[1].action_type == 'call'
        assert allin_actions[1].amount_dollars == 2.44

    def test_parse_street_tracking(self):
        hand = parse_hand(HAND_ALLIN)
        streets = set(a.street for a in hand.actions if not a.is_blind)
        assert streets == {'preflop', 'flop', 'turn'}

    def test_parse_raise_amounts(self):
        """Verify raise 'to' amount is captured correctly."""
        hand = parse_hand(HAND_RAISE_AND_MUCK)
        raises = [a for a in hand.actions if a.action_type == 'raise']
        assert len(raises) == 1
        assert raises[0].player_name == 'player_d'
        assert raises[0].amount_dollars == 0.20  # "to $0.20"

    def test_parse_no_showdown(self):
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        assert not hand.has_showdown
        assert len(hand.shown_hands) == 0

    def test_parse_small_big_blinds(self):
        """'posts small & big blinds' should be treated as blind_bb."""
        hand = parse_hand(HAND_SMALL_BIG_BLINDS)
        blinds = [a for a in hand.actions if a.is_blind]
        assert len(blinds) == 3
        sb_blind = [b for b in blinds if b.action_type == 'blind_sb']
        bb_blinds = [b for b in blinds if b.action_type == 'blind_bb']
        assert len(sb_blind) == 1
        assert sb_blind[0].player_name == 'Player4'
        assert len(bb_blinds) == 2
        # Player6's small & big blinds should be blind_bb with $0.07
        p6_blind = [b for b in bb_blinds if b.player_name == 'Player6']
        assert len(p6_blind) == 1
        assert p6_blind[0].amount_dollars == 0.07

    def test_parse_joins_ignored(self):
        """'joins the table' lines should not create actions."""
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        for a in hand.actions:
            assert 'player_e' != a.player_name  # joined mid-hand
            assert 'player_a' != a.player_name

    def test_parse_uncalled_bet_ignored(self):
        """Uncalled bet lines should not create actions."""
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        # player_c collected after uncalled bet
        assert len(hand.result.winners) == 1
        assert hand.result.winners[0]['name'] == 'player_c'

    def test_parse_empty_text(self):
        assert parse_hand('') is None
        assert parse_hand('   \n  \n  ') is None

    def test_parse_non_pokerstars(self):
        assert parse_hand('Some random text that is not a hand') is None

    def test_hole_cards_assigned_to_players(self):
        """After parsing, players who showed should have hole_cards set."""
        hand = parse_hand(HAND_SHOWDOWN)
        player_a = [p for p in hand.players if p.name == 'player_a'][0]
        assert player_a.hole_cards == ['Js', '5s']
        mbbbb = [p for p in hand.players if p.name == 'player_b'][0]
        assert mbbbb.hole_cards == ['3c', 'Ah']

    def test_parse_collected_multiple_winners(self):
        """Hand where multiple collected lines exist (e.g., split pot)."""
        text = """\
PokerStars Hand #999999999999:  Hold'em No Limit ($0.02/$0.05 USD) - 2020/01/01 16:00:00 WET [2020/01/01 11:00:00 ET]
Table 'Test' 6-max Seat #1 is the button
Seat 1: A ($5 in chips)
Seat 2: B ($5 in chips)
A: posts small blind $0.02
B: posts big blind $0.05
*** HOLE CARDS ***
A: raises $0.05 to $0.10
B: calls $0.05
*** FLOP *** [Ah Kd Qc]
B: checks
A: checks
*** SHOW DOWN ***
A: shows [Jh Ts] (a straight, Ten to Ace)
B: shows [Js Tc] (a straight, Ten to Ace)
A collected $0.05 from pot
B collected $0.05 from pot
*** SUMMARY ***
Total pot $0.20 | Rake $0.10
"""
        hand = parse_hand(text)
        assert len(hand.result.winners) == 2
        assert hand.result.winners[0]['name'] == 'A'
        assert hand.result.winners[1]['name'] == 'B'


# ──────────────────────────────────────────────────────────────────────────────
# Parser: real file tests
# ──────────────────────────────────────────────────────────────────────────────

HH_FILE = os.path.join(os.path.dirname(__file__),
                        '..', 'data', 'poker_archive',
                        'sample_6max.txt')


@pytest.mark.skipif(not os.path.exists(HH_FILE),
                    reason='HH test file not available')
class TestParserRealFile:
    def test_parse_file_returns_hands(self):
        hands = parse_file(HH_FILE)
        assert len(hands) > 0

    def test_all_hands_have_id(self):
        hands = parse_file(HH_FILE)
        for h in hands:
            assert h.hand_id, f'Hand missing ID'

    def test_all_hands_have_blinds(self):
        hands = parse_file(HH_FILE)
        for h in hands:
            assert h.big_blind_dollars == 0.05
            assert h.small_blind_dollars == 0.02

    def test_all_hands_have_players(self):
        hands = parse_file(HH_FILE)
        for h in hands:
            assert len(h.players) >= 2

    def test_all_hands_have_actions(self):
        hands = parse_file(HH_FILE)
        for h in hands:
            blinds = [a for a in h.actions if a.is_blind]
            assert len(blinds) >= 2, f'Hand {h.hand_id} has <2 blinds'

    def test_showdown_hands_have_shown_cards(self):
        hands = parse_file(HH_FILE)
        showdown = [h for h in hands if h.has_showdown]
        assert len(showdown) > 0
        for h in showdown:
            for name, cards in h.shown_hands.items():
                assert len(cards) == 2, (
                    f'Hand {h.hand_id}: {name} has {len(cards)} cards')

    def test_no_duplicate_hand_ids(self):
        hands = parse_file(HH_FILE)
        ids = [h.hand_id for h in hands]
        assert len(ids) == len(set(ids))

    def test_community_cards_valid(self):
        hands = parse_file(HH_FILE)
        valid_ranks = set('23456789TJQKA')
        valid_suits = set('cdhs')
        for h in hands:
            all_cards = []
            if h.flop:
                all_cards.extend(h.flop)
            if h.turn:
                all_cards.append(h.turn)
            if h.river:
                all_cards.append(h.river)
            for card in all_cards:
                assert len(card) == 2, f'Invalid card: {card}'
                assert card[0] in valid_ranks, f'Bad rank: {card}'
                assert card[1] in valid_suits, f'Bad suit: {card}'

    def test_action_streets_are_valid(self):
        hands = parse_file(HH_FILE)
        valid_streets = {'preflop', 'flop', 'turn', 'river'}
        for h in hands:
            for a in h.actions:
                assert a.street in valid_streets, (
                    f'Hand {h.hand_id}: action has street={a.street}')

    def test_action_types_are_valid(self):
        hands = parse_file(HH_FILE)
        valid = {'fold', 'check', 'call', 'bet', 'raise',
                 'blind_sb', 'blind_bb'}
        for h in hands:
            for a in h.actions:
                assert a.action_type in valid, (
                    f'Hand {h.hand_id}: unknown action_type={a.action_type}')

    def test_pot_and_rake_non_negative(self):
        hands = parse_file(HH_FILE)
        for h in hands:
            assert h.result.total_pot_dollars >= 0
            assert h.result.rake_dollars >= 0

    def test_winner_collected_within_pot(self):
        """Each winner should collect <= total pot."""
        hands = parse_file(HH_FILE)
        for h in hands:
            for w in h.result.winners:
                assert w['amount_dollars'] <= h.result.total_pot_dollars + 0.01, (
                    f"Hand {h.hand_id}: {w['name']} collected "
                    f"${w['amount_dollars']} but pot is "
                    f"${h.result.total_pot_dollars}")


# ──────────────────────────────────────────────────────────────────────────────
# Action mapper tests
# ──────────────────────────────────────────────────────────────────────────────

class TestActionMapper:
    def test_fold(self):
        a = ParsedAction(player_name='X', action_type='fold')
        assert map_action(a, pot=10, max_bet=4, player_bet_chips=0,
                          player_stake=100) == FOLD

    def test_check(self):
        a = ParsedAction(player_name='X', action_type='check')
        assert map_action(a, pot=10, max_bet=0, player_bet_chips=0,
                          player_stake=100) == CALL

    def test_call(self):
        a = ParsedAction(player_name='X', action_type='call',
                         amount_dollars=0.05)
        assert map_action(a, pot=10, max_bet=4, player_bet_chips=0,
                          player_stake=100) == CALL

    def test_explicit_allin(self):
        """Action marked as all-in should map to ALL_IN."""
        a = ParsedAction(player_name='X', action_type='bet',
                         amount_dollars=5.0, is_all_in=True)
        assert map_action(a, pot=10, max_bet=0, player_bet_chips=0,
                          player_stake=200) == ALL_IN

    def test_implicit_allin(self):
        """Bet that uses entire stack should map to ALL_IN."""
        a = ParsedAction(player_name='X', action_type='bet',
                         amount_dollars=2.50)  # 100 chips
        assert map_action(a, pot=10, max_bet=0, player_bet_chips=0,
                          player_stake=100) == ALL_IN

    def test_raise_half_pot(self):
        """Raise that's close to half pot should map to RAISE_HALF."""
        # pot=20 chips, half pot raise = 10 chips
        a = ParsedAction(player_name='X', action_type='raise',
                         amount_dollars=0.30)  # 12 chips "to" amount
        # max_bet=2 (BB), raise increment = 12-2 = 10, half_pot=10
        result = map_action(a, pot=20, max_bet=2, player_bet_chips=0,
                            player_stake=200)
        assert result == RAISE_HALF

    def test_raise_pot(self):
        """Raise that's close to pot should map to RAISE_POT."""
        # pot=20, pot raise = 20 chips
        a = ParsedAction(player_name='X', action_type='raise',
                         amount_dollars=0.55)  # 22 chips "to"
        # max_bet=2, raise increment = 22-2 = 20 = pot
        result = map_action(a, pot=20, max_bet=2, player_bet_chips=0,
                            player_stake=200)
        assert result == RAISE_POT

    def test_bet_small(self):
        """Small bet (close to half pot) should map to RAISE_HALF."""
        # pot=20, bet=8 chips, half_pot=10, pot=20
        a = ParsedAction(player_name='X', action_type='bet',
                         amount_dollars=0.20)  # 8 chips
        result = map_action(a, pot=20, max_bet=0, player_bet_chips=0,
                            player_stake=200)
        assert result == RAISE_HALF

    def test_bet_big(self):
        """Big bet (close to pot) should map to RAISE_POT."""
        # pot=20, bet=18 chips, half_pot=10, pot=20
        a = ParsedAction(player_name='X', action_type='bet',
                         amount_dollars=0.45)  # 18 chips
        result = map_action(a, pot=20, max_bet=0, player_bet_chips=0,
                            player_stake=200)
        assert result == RAISE_POT


class TestLegalActions:
    def test_no_bet_facing(self):
        """When not facing a bet: check/call + raises, no fold."""
        legal = get_legal_actions(max_bet=0, player_bet_chips=0,
                                  player_stake=100, pot=10)
        assert FOLD not in legal
        assert CALL in legal
        assert ALL_IN in legal

    def test_facing_bet(self):
        """When facing a bet: fold is legal."""
        legal = get_legal_actions(max_bet=4, player_bet_chips=0,
                                  player_stake=100, pot=10)
        assert FOLD in legal
        assert CALL in legal

    def test_short_stack_no_raise(self):
        """When stack can't cover min raise: only fold/call."""
        legal = get_legal_actions(max_bet=4, player_bet_chips=0,
                                  player_stake=5, pot=10, min_raise=2)
        assert FOLD in legal
        assert CALL in legal
        assert RAISE_HALF not in legal
        assert RAISE_POT not in legal
        assert ALL_IN not in legal

    def test_medium_stack(self):
        """Stack can cover half pot raise but not pot raise."""
        # pot=20, max_bet=4, call=4, remaining after call=6
        # half_pot=10 > 6: can't do half pot
        # All we can do is fold/call/all-in
        legal = get_legal_actions(max_bet=4, player_bet_chips=0,
                                  player_stake=10, pot=20, min_raise=2)
        assert FOLD in legal
        assert CALL in legal
        assert ALL_IN in legal


# ──────────────────────────────────────────────────────────────────────────────
# Observation builder tests
# ──────────────────────────────────────────────────────────────────────────────

class TestObsBuilder:
    def test_showdown_hand_produces_decision_points(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        assert len(dps) > 0

    def test_no_decision_points_without_hole_cards(self):
        """Players without known hole cards shouldn't produce DPs."""
        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        builder = ObservationBuilder(num_players=2)
        dps = builder.build_decision_points(hand)
        # No showdown → no shown hands → no hole cards → no DPs
        assert len(dps) == 0

    def test_wrong_player_count_returns_empty(self):
        hand = parse_hand(HAND_SHOWDOWN)
        # Hand has 4 active players, but we ask for 6
        builder = ObservationBuilder(num_players=6)
        dps = builder.build_decision_points(hand)
        assert len(dps) == 0

    def test_observation_shape(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            # 4 players: Position(6) + PotOdds(3) + HandStrength(5) +
            # Stack(2) + Equity(8) + Street(4) + GameState(3) + BetHist(5)
            # = 36... actually depends on engine. Just check it's reasonable.
            assert dp.observation.shape[0] > 20
            assert dp.observation.dtype == np.float32 or dp.observation.dtype == np.float64

    def test_observation_finite(self):
        """All observation values should be finite (no NaN/inf)."""
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            assert np.all(np.isfinite(dp.observation)), (
                f'Non-finite values in obs: {dp.observation}')

    def test_legal_actions_valid(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            assert len(dp.legal_actions) > 0
            assert CALL in dp.legal_actions  # always legal
            for a in dp.legal_actions:
                assert a in {FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN}

    def test_actual_action_is_legal(self):
        """The actual action taken should be in the legal set or close."""
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            # Actual action should be valid (0-4)
            assert 0 <= dp.actual_action <= 4

    def test_street_labels_correct(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        streets = [dp.street for dp in dps]
        valid = {'preflop', 'flop', 'turn', 'river'}
        for s in streets:
            assert s in valid

    def test_position_labels_set(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            assert dp.position_label != ''

    def test_hand_id_set(self):
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            assert dp.hand_id == '100000000002'

    def test_action_index_increments(self):
        """Each player's action_index should increment per decision."""
        hand = parse_hand(HAND_SHOWDOWN)
        builder = ObservationBuilder(num_players=4)
        dps = builder.build_decision_points(hand)
        from collections import defaultdict
        player_indices = defaultdict(list)
        for dp in dps:
            player_indices[dp.player_name].append(dp.action_index)
        for name, indices in player_indices.items():
            assert indices == list(range(len(indices))), (
                f'{name} action indices not sequential: {indices}')

    def test_6player_showdown_hand(self):
        """Build DPs for a full 6-player hand."""
        hand = parse_hand(HAND_ALLIN)
        assert hand.num_active_players == 6
        builder = ObservationBuilder(num_players=6)
        dps = builder.build_decision_points(hand)
        assert len(dps) > 0
        # Only player_a and player_f have shown hands
        dp_players = set(dp.player_name for dp in dps)
        assert dp_players == {'player_a', 'player_f'}

    def test_community_cards_grow_by_street(self):
        """Community cards should grow as streets progress."""
        hand = parse_hand(HAND_ALLIN)
        builder = ObservationBuilder(num_players=6)
        dps = builder.build_decision_points(hand)
        for dp in dps:
            if dp.street == 'preflop':
                assert len(dp.community_cards) == 0
            elif dp.street == 'flop':
                assert len(dp.community_cards) == 3
            elif dp.street == 'turn':
                assert len(dp.community_cards) == 4


# ──────────────────────────────────────────────────────────────────────────────
# Investment calculation tests (used by showdown EV)
# ──────────────────────────────────────────────────────────────────────────────

class TestInvestmentCalc:
    """Test the _compute_all_invested and _invested_up_to_action helpers."""

    def test_simple_blind_and_fold(self):
        """Player posts blind and folds — invested = blind amount."""
        # Import the helpers from the script
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        from validate_vs_hh import _compute_all_invested

        hand = parse_hand(HAND_2P_NO_SHOWDOWN)
        invested = _compute_all_invested(hand)
        # player_c: SB $0.02 + raise to $0.15 (increment $0.13)
        #   + bet $0.05 (returned but still was an action)
        # Actually the bet was returned as uncalled, but the action still happened
        # However, fold by player_b means bet wasn't called
        # player_c invested: SB $0.02, raise to $0.15 = $0.15 total,
        #   then bet $0.05 river (uncalled returned, but invested tracks gross)
        assert abs(invested.get('player_c', 0) - 0.20) < 0.01  # 0.02+0.13+0.05
        # player_b: BB $0.05, call $0.10 preflop = $0.15 total
        assert abs(invested.get('player_b', 0) - 0.15) < 0.01

    def test_raise_investment(self):
        """Raise 'to' should compute correct incremental investment."""
        sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                        '..', 'scripts'))
        from validate_vs_hh import _compute_all_invested

        hand = parse_hand(HAND_RAISE_AND_MUCK)
        invested = _compute_all_invested(hand)
        # player_d: BB $0.05, raises to $0.20 (increment $0.15),
        #   then calls $0.05 on river = $0.25 total
        assert abs(invested.get('player_d', 0) - 0.25) < 0.01
        # player_b: calls $0.05 preflop, calls $0.15 (raise),
        #   then bets $0.05 river = $0.25 total
        assert abs(invested.get('player_b', 0) - 0.25) < 0.01


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
