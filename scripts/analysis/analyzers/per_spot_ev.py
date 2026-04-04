"""
Per-spot EV analyzer: compare model decisions vs actual player decisions
using exact equity calculations (eval7) at showdown hands.

For each decision point where opponent cards are known, computes the
EV difference between model's preferred action and the actual play.
Results are grouped by street, hand strength, and bet size.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _PROJECT_ROOT)

import eval7

from scripts.analysis.analysis_db import AnalysisDB
from scripts.analysis.model_wrapper import ModelWrapper

# ── Constants ───────────────────────────────────────────────────────────────

ACTION_NAMES = ['Fold', 'Call', 'RaiseHalf', 'RaisePot', 'AllIn']
STREET_ORDER = ['flop', 'turn', 'river']

IDX_POT_ODDS = 10
IDX_EQUITY = 18
IDX_MADE_HAND = 19
IDX_FLUSH_DRAW = 20
IDX_STRAIGHT_DRAW = 21

HAND_CAT_THRESHOLDS = [
    (0.50, 'trips+'),
    (0.25, 'two_pair'),
    (0.115, 'pair'),
    (0.01, 'draw'),
    (0.0, 'air'),
]

BET_SIZE_CATS = [
    (0.40, 'small'),
    (0.75, 'medium'),
    (float('inf'), 'large'),
]


# ── Equity computation ──────────────────────────────────────────────────────

def _card_to_eval7(card_str):
    return eval7.Card(card_str)


def compute_equity_vs_known(hero_cards, opp_cards, community_cards):
    """Compute hero equity vs a specific opponent hand with known board."""
    try:
        hero = [_card_to_eval7(c) for c in hero_cards]
        opp = [_card_to_eval7(c) for c in opp_cards]
        board = [_card_to_eval7(c) for c in community_cards]
    except Exception:
        return 0.5

    cards_to_come = 5 - len(community_cards)

    if cards_to_come == 0:
        hero_score = eval7.evaluate(hero + board)
        opp_score = eval7.evaluate(opp + board)
        if hero_score > opp_score:
            return 1.0
        elif hero_score < opp_score:
            return 0.0
        else:
            return 0.5

    dead = set(hero + opp + board)
    deck = [c for c in eval7.Deck().cards if c not in dead]
    rng = np.random.default_rng(42)

    if cards_to_come == 1:
        sample_size = min(len(deck), 1000)
        indices = rng.choice(len(deck), size=sample_size, replace=False)
        wins = 0.0
        for idx in indices:
            full_board = board + [deck[idx]]
            h = eval7.evaluate(hero + full_board)
            o = eval7.evaluate(opp + full_board)
            if h > o:
                wins += 1.0
            elif h == o:
                wins += 0.5
        return wins / sample_size
    else:
        n_sims = 2000
        wins = 0.0
        for _ in range(n_sims):
            idxs = rng.choice(len(deck), size=2, replace=False)
            full_board = board + [deck[idxs[0]], deck[idxs[1]]]
            h = eval7.evaluate(hero + full_board)
            o = eval7.evaluate(opp + full_board)
            if h > o:
                wins += 1.0
            elif h == o:
                wins += 0.5
        return wins / n_sims


# ── Classification helpers ──────────────────────────────────────────────────

def classify_hand_strength(obs):
    made_hand = float(obs[IDX_MADE_HAND]) if len(obs) > IDX_MADE_HAND else 0.0
    has_flush_draw = float(obs[IDX_FLUSH_DRAW]) if len(obs) > IDX_FLUSH_DRAW else 0.0
    has_straight_draw = float(obs[IDX_STRAIGHT_DRAW]) if len(obs) > IDX_STRAIGHT_DRAW else 0.0

    for threshold, label in HAND_CAT_THRESHOLDS:
        if made_hand >= threshold:
            return label

    if has_flush_draw > 0.0 or has_straight_draw > 0.0:
        return 'draw'
    return 'air'


def classify_bet_size(obs):
    pot_odds = float(obs[IDX_POT_ODDS]) if len(obs) > IDX_POT_ODDS else 0.0
    if pot_odds <= 0.001:
        return 'no_bet'
    ratio = (1.0 - pot_odds) / max(pot_odds, 0.001)
    for threshold, label in BET_SIZE_CATS:
        if ratio <= threshold:
            return label
    return 'large'


# ── EV computation ──────────────────────────────────────────────────────────

def _estimate_raise_bb(action, pot_bb, obs):
    if action == 2:
        return max(pot_bb * 0.5, 1.0)
    elif action == 3:
        return pot_bb
    elif action == 4:
        return min(pot_bb * 3.0, 200.0)
    return pot_bb * 0.5


def _compute_ev_diff(model_action, actual_action, equity, pot_bb, obs):
    """Compute EV difference between model and actual action (in BB)."""
    if model_action == actual_action:
        return 0.0

    pot_odds = float(obs[IDX_POT_ODDS]) if len(obs) > IDX_POT_ODDS else 0.0
    facing_bet = pot_odds > 0.001

    if facing_bet and pot_odds > 0:
        call_cost_bb = pot_bb * (1.0 - pot_odds) / max(pot_odds, 0.001)
    else:
        call_cost_bb = 0.0

    # Fold vs Call
    if model_action == 0 and actual_action == 1:
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        return -ev_call_bb

    if model_action == 1 and actual_action == 0:
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        return ev_call_bb

    # Fold vs Raise
    if model_action == 0 and actual_action in (2, 3, 4):
        raise_amount_bb = _estimate_raise_bb(actual_action, pot_bb, obs)
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return -ev_raise_bb

    if model_action in (2, 3, 4) and actual_action == 0:
        raise_amount_bb = _estimate_raise_bb(model_action, pot_bb, obs)
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_raise_bb

    # Call vs Raise
    if model_action == 1 and actual_action in (2, 3, 4):
        raise_amount_bb = _estimate_raise_bb(actual_action, pot_bb, obs)
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_call_bb - ev_raise_bb

    if model_action in (2, 3, 4) and actual_action == 1:
        raise_amount_bb = _estimate_raise_bb(model_action, pot_bb, obs)
        ev_call_bb = equity * (pot_bb + call_cost_bb) - call_cost_bb
        new_pot_bb = pot_bb + raise_amount_bb + call_cost_bb
        ev_raise_bb = equity * new_pot_bb - raise_amount_bb
        return ev_raise_bb - ev_call_bb

    # Raise size differences
    if model_action in (2, 3, 4) and actual_action in (2, 3, 4):
        model_raise = _estimate_raise_bb(model_action, pot_bb, obs)
        actual_raise = _estimate_raise_bb(actual_action, pot_bb, obs)
        model_pot = pot_bb + model_raise + call_cost_bb
        actual_pot = pot_bb + actual_raise + call_cost_bb
        ev_model = equity * model_pot - model_raise
        ev_actual = equity * actual_pot - actual_raise
        return ev_model - ev_actual

    return 0.0


# ── Spot result container ──────────────────────────────────────────────────

class SpotResult:
    __slots__ = ('street', 'hand_cat', 'bet_size', 'model_action',
                 'actual_action', 'ev_diff_bb', 'equity', 'pot_bb',
                 'hand_id', 'hero_cards', 'opp_cards', 'position',
                 'model_strategy')

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


# ── Main analysis function ──────────────────────────────────────────────────

def analyze(
    db: AnalysisDB,
    model: ModelWrapper,
    **kwargs,
) -> dict:
    """Run per-spot EV analysis comparing model vs actual play.

    Requires hands with 2+ shown cards for precise equity calculation.

    Returns:
        Dictionary with all analysis results.
    """
    # Build shown_hands lookup by hand_id
    hand_shown = {}
    for hand in db.hands:
        if len(hand.shown_hands) >= 2:
            hand_shown[hand.hand_id] = hand.shown_hands

    print(f'  Hands with 2+ shown cards: {len(hand_shown)}')

    all_results = []
    n_processed = 0

    for dp in db.decision_points:
        # Skip preflop
        if dp.street == 'preflop':
            continue
        # Need shown cards for this player
        if dp.hand_id not in hand_shown:
            continue
        shown = hand_shown[dp.hand_id]
        if dp.player_name not in shown:
            continue
        if not dp.hole_cards:
            continue

        hero_cards = dp.hole_cards
        opp_names = [n for n in shown if n != dp.player_name]
        if not opp_names:
            continue

        opp_cards = shown[opp_names[0]]

        # Compute equity vs opponent's actual hand
        equity = compute_equity_vs_known(hero_cards, opp_cards, dp.community_cards)

        # Get model decision
        try:
            m_action, strategy = model.get_action(dp.observation, dp.legal_actions)
        except Exception:
            continue

        bb = 0.05
        pot_bb = dp.pot_dollars / bb
        hand_cat = classify_hand_strength(dp.observation)
        bet_size = classify_bet_size(dp.observation)

        ev_diff = _compute_ev_diff(
            m_action, dp.actual_action, equity, pot_bb, dp.observation)

        all_results.append(SpotResult(
            street=dp.street,
            hand_cat=hand_cat,
            bet_size=bet_size,
            model_action=m_action,
            actual_action=dp.actual_action,
            ev_diff_bb=ev_diff,
            equity=equity,
            pot_bb=pot_bb,
            hand_id=dp.hand_id,
            hero_cards=hero_cards,
            opp_cards=opp_cards,
            position=dp.position_label,
            model_strategy=strategy,
        ))

        n_processed += 1
        if n_processed % 5000 == 0:
            print(f'    {n_processed} decision points analyzed...', flush=True)

    # Group by (street, hand_cat, bet_size)
    grouped = defaultdict(list)
    for r in all_results:
        key = (r.street, r.hand_cat, r.bet_size)
        grouped[key].append(r)

    return {
        'all_results': all_results,
        'grouped': dict(grouped),
        'total_hands': len(hand_shown),
    }


# ── Report printing ─────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print the per-spot EV analysis report."""
    all_results = results['all_results']
    grouped = results['grouped']
    total_hands = results['total_hands']

    print('\n' + '=' * 90)
    print('PER-SPOT EV ANALYSIS')
    print('=' * 90)
    print(f'Total hands analyzed: {total_hands}')
    print(f'Total decision points: {len(all_results)}')

    disagreements = [r for r in all_results if r.model_action != r.actual_action]
    agreements = [r for r in all_results if r.model_action == r.actual_action]
    n = max(len(all_results), 1)
    print(f'Agreements: {len(agreements)} ({100*len(agreements)/n:.1f}%)')
    print(f'Disagreements: {len(disagreements)} ({100*len(disagreements)/n:.1f}%)')

    if disagreements:
        total_ev = sum(r.ev_diff_bb for r in disagreements)
        print(f'Total EV difference (disagreements): {total_ev:+.2f} BB')

    # Per-street summary
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

    # Per hand-strength summary
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

    # Per bet-size summary
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

    # Detailed per-spot table
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

    # Top disagreements
    print('\n' + '-' * 90)
    print('TOP 15 DISAGREEMENTS BY |EV DIFFERENCE|')
    print('-' * 90)
    print(f'{"Hand":>12} {"Street":<6} {"Pos":<4} {"Hero":<8} {"Opp":<8} '
          f'{"Eq":>5} {"Pot":>6} {"Model":>10} {"Actual":>10} {"EV Diff":>8}')
    print('-' * 90)

    sorted_d = sorted(disagreements, key=lambda r: abs(r.ev_diff_bb), reverse=True)
    for r in sorted_d[:15]:
        hero_str = ''.join(r.hero_cards) if r.hero_cards else '??'
        opp_str = ''.join(r.opp_cards) if r.opp_cards else '??'
        print(f'#{r.hand_id:>11} {r.street:<6} {r.position:<4} {hero_str:<8} {opp_str:<8} '
              f'{r.equity:>4.2f} {r.pot_bb:>5.1f} {ACTION_NAMES[r.model_action]:>10} '
              f'{ACTION_NAMES[r.actual_action]:>10} {r.ev_diff_bb:>+7.2f}')

    # BB/100 projection
    print('\n' + '-' * 90)
    print('BB/100 PROJECTION')
    print('-' * 90)

    if disagreements and total_hands > 0:
        total_ev = sum(r.ev_diff_bb for r in disagreements)
        ev_per_hand = total_ev / total_hands
        bb_per_100 = ev_per_hand * 100
        print(f'Postflop EV improvement (model vs field): {bb_per_100:+.2f} BB/100')
        print(f'  Total EV diff: {total_ev:+.2f} BB over {total_hands} hands')
        print(f'  Per-hand: {ev_per_hand:+.4f} BB')

        for street in STREET_ORDER:
            street_d = [r for r in disagreements if r.street == street]
            if street_d:
                s_ev = sum(r.ev_diff_bb for r in street_d)
                s_bb100 = (s_ev / total_hands) * 100
                print(f'  {street}: {s_bb100:+.2f} BB/100 ({len(street_d)} disagreements)')
    else:
        print('No disagreements found.')

    # Action distribution
    print('\n' + '-' * 90)
    print('ACTION DISTRIBUTION: MODEL vs ACTUAL')
    print('-' * 90)
    print(f'{"Action":<12} {"Model%":>8} {"Actual%":>8} {"Diff":>8}')
    print('-' * 50)

    for i, name in enumerate(ACTION_NAMES):
        m_pct = 100 * sum(1 for r in all_results if r.model_action == i) / n
        a_pct = 100 * sum(1 for r in all_results if r.actual_action == i) / n
        print(f'{name:<12} {m_pct:>7.1f}% {a_pct:>7.1f}% {m_pct - a_pct:>+7.1f}%')

    print('\n' + '=' * 90)
