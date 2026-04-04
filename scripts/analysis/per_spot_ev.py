"""Per-spot EV analysis: compare model decisions vs actual player decisions.

Uses showdown hand histories where both players' cards are known to compute
precise EV differences between model strategy and actual play.

Usage:
  python3 scripts/analysis/per_spot_ev.py \
      --model-facing-bet models/best/cql_6p_facing_bet_v7.pt \
      --model-checked-to models/best/cql_6p_checked_to_v3.pt \
      --max-hands 20000
"""

import sys
import os
import argparse
import time
import multiprocessing as mp
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────────
project_root = str(Path(__file__).resolve().parent.parent.parent)
sys.path.insert(0, project_root)

import torch
import eval7

from scripts.train_cql import QNetwork, CQLModelWrapper
from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.models import ParsedHand, dollars_to_chips


# ── Constants ────────────────────────────────────────────────────────────────

ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn']
STREET_ORDER = ['flop', 'turn', 'river']

# Observation feature indices (50-dim with behavioral stats)
IDX_POT_ODDS = 10      # pot_odds (0 when not facing a bet)
IDX_EQUITY = 18         # hand_strength / equity vs random
IDX_MADE_HAND = 19      # made_hand_cat (0=air, 0.115=pair, 0.25=2pair, etc.)
IDX_FLUSH_DRAW = 20
IDX_STRAIGHT_DRAW = 21

# Hand strength categories (made_hand_cat thresholds from obs builder)
HAND_CAT_THRESHOLDS = [
    (0.50, 'trips+'),
    (0.25, 'two_pair'),
    (0.115, 'pair'),
    (0.01, 'draw'),
    (0.0, 'air'),
]

# Bet size categories (call_amount / pot ratio)
BET_SIZE_CATS = [
    (0.40, 'small'),    # <= 40% pot
    (0.75, 'medium'),   # <= 75% pot
    (float('inf'), 'large'),  # > 75% pot
]


# ── Model loading ────────────────────────────────────────────────────────────

def load_cql_model(path, device='cpu'):
    """Load a CQL model checkpoint and return (wrapper, obs_size)."""
    ckpt = torch.load(path, map_location=device)
    obs_size = ckpt['obs_size']
    action_size = ckpt['action_size']
    hidden = ckpt.get('hidden_layers', [512, 512, 256])
    dropout = ckpt.get('dropout', 0.1)
    temperature = ckpt.get('temperature', 1.0)
    num_players = ckpt.get('num_players', 6)

    q_net = QNetwork(obs_size, action_size, hidden, dropout)
    q_net.load_state_dict(ckpt['model_state_dict'])
    q_net.eval()

    wrapper = CQLModelWrapper(
        q_net=q_net,
        obs_size=obs_size,
        action_size=action_size,
        num_players=num_players,
        device=device,
        temperature=temperature,
    )
    return wrapper, obs_size


# ── Equity computation using eval7 ──────────────────────────────────────────

def _card_str_to_eval7(card_str):
    """Convert PokerStars card string (e.g. 'Ah') to eval7.Card."""
    return eval7.Card(card_str)


def compute_equity_vs_known(hero_cards, opp_cards, community_cards):
    """Compute hero equity vs a specific opponent hand with known board.

    For river (5 community cards): exact evaluation.
    For flop/turn: Monte Carlo over remaining cards.

    Returns: float equity in [0, 1].
    """
    hero = [_card_str_to_eval7(c) for c in hero_cards]
    opp = [_card_str_to_eval7(c) for c in opp_cards]
    board = [_card_str_to_eval7(c) for c in community_cards]

    cards_to_come = 5 - len(community_cards)

    if cards_to_come == 0:
        # River: exact evaluation (higher score = better hand in eval7)
        hero_score = eval7.evaluate(hero + board)
        opp_score = eval7.evaluate(opp + board)
        if hero_score > opp_score:
            return 1.0
        elif hero_score < opp_score:
            return 0.0
        else:
            return 0.5

    # Flop or turn: Monte Carlo over remaining cards
    dead = set(hero + opp + board)
    deck = [c for c in eval7.Deck().cards if c not in dead]

    n_sims = 2000 if cards_to_come == 2 else 1000  # more sims for flop
    wins = 0.0
    count = 0

    if cards_to_come == 1:
        # Turn -> River: enumerate or sample single card
        sample_size = min(len(deck), n_sims)
        rng = np.random.default_rng(42)
        indices = rng.choice(len(deck), size=sample_size, replace=False)
        for idx in indices:
            full_board = board + [deck[idx]]
            h = eval7.evaluate(hero + full_board)
            o = eval7.evaluate(opp + full_board)
            if h > o:
                wins += 1.0
            elif h == o:
                wins += 0.5
            count += 1
    else:
        # Flop -> Turn+River: sample pairs of cards
        rng = np.random.default_rng(42)
        for _ in range(n_sims):
            idxs = rng.choice(len(deck), size=2, replace=False)
            full_board = board + [deck[idxs[0]], deck[idxs[1]]]
            h = eval7.evaluate(hero + full_board)
            o = eval7.evaluate(opp + full_board)
            if h > o:
                wins += 1.0
            elif h == o:
                wins += 0.5
            count += 1

    return wins / max(count, 1)


# ── Hand strength category ──────────────────────────────────────────────────

def classify_hand_strength(obs):
    """Classify hand strength from observation features."""
    made_hand = float(obs[IDX_MADE_HAND]) if len(obs) > IDX_MADE_HAND else 0.0
    has_flush_draw = float(obs[IDX_FLUSH_DRAW]) if len(obs) > IDX_FLUSH_DRAW else 0.0
    has_straight_draw = float(obs[IDX_STRAIGHT_DRAW]) if len(obs) > IDX_STRAIGHT_DRAW else 0.0

    for threshold, label in HAND_CAT_THRESHOLDS:
        if made_hand >= threshold:
            return label

    # Check draws if air
    if has_flush_draw > 0.0 or has_straight_draw > 0.0:
        return 'draw'
    return 'air'


def classify_bet_size(obs):
    """Classify bet size from pot_odds in observation.

    pot_odds = pot / (pot + call_amount), so:
    call_amount/pot = (1 - pot_odds) / pot_odds
    """
    pot_odds = float(obs[IDX_POT_ODDS]) if len(obs) > IDX_POT_ODDS else 0.0
    if pot_odds <= 0.001:
        return 'no_bet'  # checked to
    # call_amount / pot ratio
    ratio = (1.0 - pot_odds) / max(pot_odds, 0.001)
    for threshold, label in BET_SIZE_CATS:
        if ratio <= threshold:
            return label
    return 'large'


# ── Model decision ───────────────────────────────────────────────────────────

def get_model_action(dp, model_facing_bet, model_checked_to, obs_size_fb, obs_size_ct, model_fb_3bet=None, obs_size_fb3=None):
    """Get the model's preferred action for a decision point.

    Uses facing_bet model when pot_odds > 0, else checked_to model
    (with fold removed from legal actions).

    Returns: (chosen_action, strategy_array)
    """
    obs = dp.observation
    pot_odds = float(obs[IDX_POT_ODDS]) if len(obs) > IDX_POT_ODDS else 0.0

    if pot_odds > 0.001:
        # Facing a bet: use facing_bet model, all legal actions
        pot_bb = obs[8] * 600 if len(obs) > 8 else 0
        if pot_bb > 12 and model_fb_3bet is not None:
            model = model_fb_3bet
            target_size = obs_size_fb3 or obs_size_fb
        else:
            model = model_facing_bet
        legal = list(dp.legal_actions)
    else:
        # Checked to: use checked_to model, fold removed
        model = model_checked_to
        legal = [a for a in dp.legal_actions if a != 0]
        if not legal:
            legal = [1]  # fallback to call/check

    strategy = model.get_strategy(obs, dp.player_idx, legal)
    chosen = int(np.argmax(strategy))
    return chosen, strategy


# ── Spot result ──────────────────────────────────────────────────────────────

class SpotResult:
    """One decision point comparison between model and actual player."""
    __slots__ = ('street', 'hand_cat', 'bet_size', 'model_action',
                 'actual_action', 'ev_diff_bb', 'equity', 'pot_bb',
                 'hand_id', 'hero_cards', 'opp_cards', 'position',
                 'model_strategy')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Main analysis ────────────────────────────────────────────────────────────

def analyze_hand(hand, builder, model_fb, model_ct, obs_size_fb, obs_size_ct, model_fb_3bet=None, obs_size_fb3=None):
    """Analyze all postflop decision points in a showdown hand.

    Returns list of SpotResult for each disagreement and agreement.
    """
    results = []

    # Need exactly 2 shown hands for precise equity calculation
    shown_players = list(hand.shown_hands.keys())
    if len(shown_players) < 2:
        return results

    # Build decision points with behavioral stats
    dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                        behavioral_stats=True)
    if not dps:
        return results

    # Compute net profit per player
    player_winnings = {}
    for w in hand.result.winners:
        name = w['name']
        player_winnings[name] = player_winnings.get(name, 0) + w['amount_dollars']

    player_invested = defaultdict(float)
    round_committed = defaultdict(float)
    current_street = 'preflop'
    for action in hand.actions:
        if action.street != current_street:
            current_street = action.street
            round_committed.clear()
        name = action.player_name
        if action.is_blind or action.action_type in ('bet', 'call'):
            player_invested[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'raise':
            to_amt = action.amount_dollars
            already = round_committed.get(name, 0)
            delta = to_amt - already
            if delta > 0:
                player_invested[name] += delta
            round_committed[name] = to_amt

    bb = hand.big_blind_dollars

    # For each shown player pair, analyze postflop decision points
    for dp in dps:
        if dp.street == 'preflop':
            continue
        if dp.player_name not in hand.shown_hands:
            continue
        if not dp.hole_cards:
            continue

        hero_name = dp.player_name
        hero_cards = dp.hole_cards

        # Find opponent(s) with known cards
        opp_names = [n for n in shown_players if n != hero_name]
        if not opp_names:
            continue

        # Use first opponent with known cards for equity calc
        opp_name = opp_names[0]
        opp_cards = hand.shown_hands[opp_name]

        # Compute equity vs opponent's actual hand
        equity = compute_equity_vs_known(hero_cards, opp_cards, dp.community_cards)

        # Get model decision
        model_action, strategy = get_model_action(
            dp, model_fb, model_ct, obs_size_fb, obs_size_ct, model_fb_3bet, obs_size_fb3)

        actual_action = dp.actual_action
        pot_bb = dp.pot_dollars / bb
        hand_cat = classify_hand_strength(dp.observation)
        bet_size = classify_bet_size(dp.observation)

        # Compute EV difference in BB
        ev_diff = _compute_ev_diff(
            model_action=model_action,
            actual_action=actual_action,
            equity=equity,
            pot_bb=pot_bb,
            dp=dp,
            hand=hand,
            hero_name=hero_name,
            player_invested=player_invested,
            player_winnings=player_winnings,
            bb=bb,
        )

        results.append(SpotResult(
            street=dp.street,
            hand_cat=hand_cat,
            bet_size=bet_size,
            model_action=model_action,
            actual_action=actual_action,
            ev_diff_bb=ev_diff,
            equity=equity,
            pot_bb=pot_bb,
            hand_id=hand.hand_id,
            hero_cards=hero_cards,
            opp_cards=opp_cards,
            position=dp.position_label,
            model_strategy=strategy,
        ))

    return results


def _compute_ev_diff(model_action, actual_action, equity, pot_bb, dp, hand,
                     hero_name, player_invested, player_winnings, bb):
    """Compute EV difference between model and actual action (in BB).

    Positive = model is better, negative = actual play was better.
    """
    if model_action == actual_action:
        return 0.0

    pot_odds = float(dp.observation[IDX_POT_ODDS]) if len(dp.observation) > IDX_POT_ODDS else 0.0
    facing_bet = pot_odds > 0.001

    # Compute call cost in BB
    if facing_bet and pot_odds > 0:
        # call_cost / pot = (1 - pot_odds) / pot_odds
        call_cost_bb = pot_bb * (1.0 - pot_odds) / max(pot_odds, 0.001)
    else:
        call_cost_bb = 0.0

    won = player_winnings.get(hero_name, 0.0) / bb
    invested = player_invested.get(hero_name, 0.0) / bb
    net_bb = won - invested

    # ── Fold vs Call disagreements ────────────────────────────────────
    if model_action == 0 and actual_action == 1:
        # Model folds, player called
        # Player's actual outcome from calling is known from hand result
        # But we need the EV of calling at THIS point, not the final hand result
        # EV(call) = equity * (pot + call_cost) - (1 - equity) * call_cost
        #          = equity * pot - call_cost * (1 - 2*equity)
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        ev_fold = 0.0  # fold has 0 marginal EV from this point
        # Model saves: ev_fold - ev_call = -ev_call (positive if call was -EV)
        return ev_fold - ev_call_bb

    if model_action == 1 and actual_action == 0:
        # Model calls, player folded
        # EV of calling based on equity
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        ev_fold = 0.0
        return ev_call_bb - ev_fold

    # ── Fold vs Raise disagreements ──────────────────────────────────
    if model_action == 0 and actual_action in (2, 3, 4):
        # Model folds, player raised: estimate raise EV pessimistically
        # Use equity-based EV with pot + raise sizing
        raise_amount_bb = _estimate_raise_bb(actual_action, pot_bb, dp)
        # Assume opponent calls the raise (worst case for raiser)
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return -ev_raise_bb  # model saves by not raising

    if model_action in (2, 3, 4) and actual_action == 0:
        # Model raises, player folded
        raise_amount_bb = _estimate_raise_bb(model_action, pot_bb, dp)
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_raise_bb  # model captures value by raising

    # ── Call vs Raise disagreements ──────────────────────────────────
    if model_action == 1 and actual_action in (2, 3, 4):
        # Model calls, player raised
        raise_amount_bb = _estimate_raise_bb(actual_action, pot_bb, dp)
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_call_bb - ev_raise_bb

    if model_action in (2, 3, 4) and actual_action == 1:
        # Model raises, player called
        raise_amount_bb = _estimate_raise_bb(model_action, pot_bb, dp)
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_raise_bb - ev_call_bb

    # ── Raise size disagreements ─────────────────────────────────────
    if model_action in (2, 3, 4) and actual_action in (2, 3, 4):
        model_raise = _estimate_raise_bb(model_action, pot_bb, dp)
        actual_raise = _estimate_raise_bb(actual_action, pot_bb, dp)
        # EV difference from sizing
        model_pot = pot_bb + model_raise + call_cost_bb
        actual_pot = pot_bb + actual_raise + call_cost_bb
        ev_model = equity * model_pot - model_raise
        ev_actual = equity * actual_pot - actual_raise
        return ev_model - ev_actual

    return 0.0


def _estimate_raise_bb(action, pot_bb, dp):
    """Estimate raise size in BB for a given action."""
    if action == 2:  # raise half pot
        return max(pot_bb * 0.5, 1.0)
    elif action == 3:  # raise pot
        return pot_bb
    elif action == 4:  # all-in
        # Use stack size from observation
        obs = dp.observation
        # stack_to_pot at index 9, pot_normalized at index 8
        # Approximate: use pot * 5 as max all-in (capped)
        return min(pot_bb * 3.0, 200.0)  # conservative estimate
    return pot_bb * 0.5  # fallback


# ── Aggregation and display ──────────────────────────────────────────────────

def aggregate_results(all_results):
    """Aggregate SpotResults into per-spot EV tables."""
    # Group by (street, hand_cat, bet_size)
    grouped = defaultdict(list)
    for r in all_results:
        key = (r.street, r.hand_cat, r.bet_size)
        grouped[key].append(r)

    return grouped


def print_summary(grouped, all_results, total_hands):
    """Print per-spot EV table and summary statistics."""
    print('\n' + '=' * 90)
    print('PER-SPOT EV ANALYSIS')
    print('=' * 90)
    print(f'Total hands analyzed: {total_hands}')
    print(f'Total decision points: {len(all_results)}')

    disagreements = [r for r in all_results if r.model_action != r.actual_action]
    agreements = [r for r in all_results if r.model_action == r.actual_action]
    print(f'Agreements: {len(agreements)} ({100*len(agreements)/max(len(all_results),1):.1f}%)')
    print(f'Disagreements: {len(disagreements)} ({100*len(disagreements)/max(len(all_results),1):.1f}%)')

    if disagreements:
        total_ev = sum(r.ev_diff_bb for r in disagreements)
        print(f'Total EV difference (disagreements): {total_ev:+.2f} BB')

    # ── Per-street summary ───────────────────────────────────────────────
    print('\n' + '-' * 90)
    print(f'{"Street":<8} {"Spots":>6} {"Disagree":>9} {"Agree%":>7} '
          f'{"EV Diff":>9} {"Avg EV":>8} {"Avg Equity":>10}')
    print('-' * 90)

    for street in STREET_ORDER:
        spots = [r for r in all_results if r.street == street]
        if not spots:
            continue
        disagree = [r for r in spots if r.model_action != r.actual_action]
        agree_pct = 100 * (len(spots) - len(disagree)) / len(spots)
        ev_sum = sum(r.ev_diff_bb for r in disagree)
        avg_ev = ev_sum / max(len(disagree), 1)
        avg_eq = np.mean([r.equity for r in spots])
        print(f'{street:<8} {len(spots):>6} {len(disagree):>9} {agree_pct:>6.1f}% '
              f'{ev_sum:>+8.2f} {avg_ev:>+7.3f} {avg_eq:>10.3f}')

    # ── Per hand-strength summary ────────────────────────────────────────
    print('\n' + '-' * 90)
    print(f'{"Hand Cat":<10} {"Spots":>6} {"Disagree":>9} {"Agree%":>7} '
          f'{"EV Diff":>9} {"Avg EV":>8} {"Model Fold%":>11} {"Actual Fold%":>12}')
    print('-' * 90)

    for cat in ['trips+', 'two_pair', 'pair', 'draw', 'air']:
        spots = [r for r in all_results if r.hand_cat == cat]
        if not spots:
            continue
        disagree = [r for r in spots if r.model_action != r.actual_action]
        agree_pct = 100 * (len(spots) - len(disagree)) / len(spots)
        ev_sum = sum(r.ev_diff_bb for r in disagree)
        avg_ev = ev_sum / max(len(disagree), 1)
        model_fold_pct = 100 * sum(1 for r in spots if r.model_action == 0) / len(spots)
        actual_fold_pct = 100 * sum(1 for r in spots if r.actual_action == 0) / len(spots)
        print(f'{cat:<10} {len(spots):>6} {len(disagree):>9} {agree_pct:>6.1f}% '
              f'{ev_sum:>+8.2f} {avg_ev:>+7.3f} {model_fold_pct:>10.1f}% {actual_fold_pct:>11.1f}%')

    # ── Per bet-size summary ─────────────────────────────────────────────
    print('\n' + '-' * 90)
    print(f'{"Bet Size":<10} {"Spots":>6} {"Disagree":>9} {"Agree%":>7} '
          f'{"EV Diff":>9} {"Avg EV":>8}')
    print('-' * 90)

    for bsize in ['no_bet', 'small', 'medium', 'large']:
        spots = [r for r in all_results if r.bet_size == bsize]
        if not spots:
            continue
        disagree = [r for r in spots if r.model_action != r.actual_action]
        agree_pct = 100 * (len(spots) - len(disagree)) / len(spots)
        ev_sum = sum(r.ev_diff_bb for r in disagree)
        avg_ev = ev_sum / max(len(disagree), 1)
        print(f'{bsize:<10} {len(spots):>6} {len(disagree):>9} {agree_pct:>6.1f}% '
              f'{ev_sum:>+8.2f} {avg_ev:>+7.3f}')

    # ── Detailed per-spot table ──────────────────────────────────────────
    print('\n' + '-' * 90)
    print('DETAILED SPOT BREAKDOWN (street / hand strength / bet size)')
    print('-' * 90)
    print(f'{"Street":<7} {"HandCat":<10} {"BetSize":<8} {"N":>5} {"Disagree":>9} '
          f'{"EV Diff":>9} {"AvgEq":>6} {"M->Fold":>8} {"M->Call":>8} {"M->Raise":>9}')
    print('-' * 90)

    for street in STREET_ORDER:
        for cat in ['trips+', 'two_pair', 'pair', 'draw', 'air']:
            for bsize in ['no_bet', 'small', 'medium', 'large']:
                key = (street, cat, bsize)
                spots = grouped.get(key, [])
                if len(spots) < 3:
                    continue
                disagree = [r for r in spots if r.model_action != r.actual_action]
                ev_sum = sum(r.ev_diff_bb for r in disagree)
                avg_eq = np.mean([r.equity for r in spots])
                m_fold = 100 * sum(1 for r in spots if r.model_action == 0) / len(spots)
                m_call = 100 * sum(1 for r in spots if r.model_action == 1) / len(spots)
                m_raise = 100 * sum(1 for r in spots if r.model_action in (2, 3, 4)) / len(spots)
                print(f'{street:<7} {cat:<10} {bsize:<8} {len(spots):>5} {len(disagree):>9} '
                      f'{ev_sum:>+8.2f} {avg_eq:>5.3f} {m_fold:>7.1f}% {m_call:>7.1f}% {m_raise:>8.1f}%')

    # ── Biggest disagreements (top 10 spots by |EV diff|) ───────────────
    print('\n' + '-' * 90)
    print('TOP 15 DISAGREEMENTS BY |EV DIFFERENCE|')
    print('-' * 90)
    print(f'{"Hand":>12} {"Street":<6} {"Pos":<4} {"Hero":<8} {"Opp":<8} '
          f'{"Eq":>5} {"Pot":>6} {"Model":>10} {"Actual":>10} {"EV Diff":>8}')
    print('-' * 90)

    sorted_d = sorted(disagreements, key=lambda r: abs(r.ev_diff_bb), reverse=True)
    for r in sorted_d[:15]:
        hero_str = ''.join(r.hero_cards)
        opp_str = ''.join(r.opp_cards)
        print(f'#{r.hand_id:>11} {r.street:<6} {r.position:<4} {hero_str:<8} {opp_str:<8} '
              f'{r.equity:>4.2f} {r.pot_bb:>5.1f} {ACTION_NAMES[r.model_action]:>10} '
              f'{ACTION_NAMES[r.actual_action]:>10} {r.ev_diff_bb:>+7.2f}')

    # ── BB/100 projection ────────────────────────────────────────────────
    print('\n' + '-' * 90)
    print('BB/100 PROJECTION')
    print('-' * 90)

    if disagreements:
        total_ev = sum(r.ev_diff_bb for r in disagreements)
        # Average EV per hand = total_ev / hands
        # BB/100 = (total_ev / total_hands) * 100
        if total_hands > 0:
            ev_per_hand = total_ev / total_hands
            bb_per_100 = ev_per_hand * 100
            print(f'Postflop EV improvement (model vs field): {bb_per_100:+.2f} BB/100')
            print(f'  Total EV diff: {total_ev:+.2f} BB over {total_hands} hands')
            print(f'  Per-hand: {ev_per_hand:+.4f} BB')

            # Break down by street
            for street in STREET_ORDER:
                street_d = [r for r in disagreements if r.street == street]
                if street_d:
                    s_ev = sum(r.ev_diff_bb for r in street_d)
                    s_bb100 = (s_ev / total_hands) * 100
                    print(f'  {street}: {s_bb100:+.2f} BB/100 ({len(street_d)} disagreements)')
    else:
        print('No disagreements found.')

    # ── Action distribution comparison ───────────────────────────────────
    print('\n' + '-' * 90)
    print('ACTION DISTRIBUTION: MODEL vs ACTUAL')
    print('-' * 90)
    print(f'{"Action":<12} {"Model%":>8} {"Actual%":>8} {"Diff":>8}')
    print('-' * 50)

    n = max(len(all_results), 1)
    for i, name in enumerate(ACTION_NAMES):
        m_pct = 100 * sum(1 for r in all_results if r.model_action == i) / n
        a_pct = 100 * sum(1 for r in all_results if r.actual_action == i) / n
        print(f'{name:<12} {m_pct:>7.1f}% {a_pct:>7.1f}% {m_pct - a_pct:>+7.1f}%')

    print('\n' + '=' * 90)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Per-spot EV analysis')
    parser.add_argument('--model-facing-bet', required=True,
                        help='Path to CQL model for facing-bet spots')
    parser.add_argument('--model-checked-to', required=True,
                        help='Path to CQL model for checked-to spots')
    parser.add_argument('--model-facing-bet-3bet', type=str, default=None,
                        help='Model for facing-bet spots in 3bet pots (pot > 12BB)')
    parser.add_argument('--hh-dir', default='data/poker_archive',
                        help='Hand history directory')
    parser.add_argument('--max-hands', type=int, default=20000,
                        help='Maximum showdown hands to analyze')
    parser.add_argument('--num-players', type=int, default=6,
                        help='Number of players per table')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for model inference')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress for every 1000 hands')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parsing workers (0=auto)')
    args = parser.parse_args()

    t0 = time.time()

    # Find HH files
    hh_dir = Path(project_root) / args.hh_dir
    hh_files = sorted(hh_dir.rglob('*.txt'))
    print(f'Found {len(hh_files)} HH files in {hh_dir}')

    # Parse FIRST (before torch/CUDA — mp.Pool deadlocks after CUDA init)
    from scripts.analysis._parse_utils import parallel_parse
    t_parse = time.time()
    all_hands = parallel_parse(hh_files, max_hands=args.max_hands,
                                workers=getattr(args, 'workers', 0))
    all_hands = [h for h in all_hands if len(h.shown_hands) >= 2]
    print(f'  Hands with 2+ shown cards: {len(all_hands)} ({time.time()-t_parse:.1f}s)')

    # NOW load models (after mp.Pool is done)
    print(f'Loading models...')
    model_fb, obs_size_fb = load_cql_model(args.model_facing_bet, args.device)
    model_ct, obs_size_ct = load_cql_model(args.model_checked_to, args.device)
    print(f'  facing_bet: obs_size={obs_size_fb}')
    print(f'  checked_to: obs_size={obs_size_ct}')
    model_fb_3bet = None
    obs_size_fb3 = None
    if args.model_facing_bet_3bet:
        model_fb_3bet, obs_size_fb3 = load_cql_model(args.model_facing_bet_3bet, args.device)
        print(f'  3bet facing-bet: obs_size={obs_size_fb3}')

    builder = ObservationBuilder(num_players=args.num_players)

    all_results = []
    total_showdown = len(all_hands)
    skipped_no_cards = 0

    for hi, hand in enumerate(all_hands):
        results = analyze_hand(hand, builder, model_fb, model_ct,
                               obs_size_fb, obs_size_ct,
                               model_fb_3bet, obs_size_fb3)
        all_results.extend(results)
        if (hi + 1) % 2000 == 0:
            print(f'  [{hi+1}/{len(all_hands)}] {len(all_results)} decisions...')

        if args.verbose and fi % 100 == 0 and fi > 0:
            elapsed = time.time() - t0
            print(f'  [{fi}/{len(hh_files)}] {total_showdown} showdown hands, '
                  f'{len(all_results)} decision points, {elapsed:.1f}s')

    elapsed = time.time() - t0
    print(f'\nParsed {total_showdown} showdown hands '
          f'({skipped_no_cards} skipped <2 shown)')
    print(f'Collected {len(all_results)} postflop decision points in {elapsed:.1f}s')

    if not all_results:
        print('No results to analyze.')
        return

    # Aggregate and display
    grouped = aggregate_results(all_results)
    print_summary(grouped, all_results, total_showdown)




if __name__ == '__main__':
    main()
