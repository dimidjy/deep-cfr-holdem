"""
Benchmark analyzer: compare model decisions against known winning players.

Finds postflop showdown hands where good players acted, computes what the
model would do at each decision point, and reports agreement rates grouped
by winner win-rate tier and spot type.
"""

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np

_PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.analysis_db import AnalysisDB, compute_player_winrates
from scripts.analysis.model_wrapper import ModelWrapper

# ── Constants ───────────────────────────────────────────────────────────────

FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4
ACTION_NAMES = ['Fold', 'Call', 'Raise1/2', 'RaisePot', 'AllIn']

IDX_POT_ODDS = 10
IDX_MADE_HAND = 19

TIERS = [
    (5.0, 10.0, '5-10 BB/100'),
    (10.0, 15.0, '10-15 BB/100'),
    (15.0, 999.0, '15+ BB/100'),
]

SPOT_NAMES = {
    'facing_bet_air': 'Facing bet, air (high card)',
    'facing_bet_pair': 'Facing bet, one pair',
    'facing_bet_strong': 'Facing bet, two pair+',
    'facing_bet_draw': 'Facing bet, draw',
    'checked_to_air': 'Checked to, air (high card)',
    'checked_to_pair': 'Checked to, one pair',
    'checked_to_strong': 'Checked to, two pair+',
    'checked_to_draw': 'Checked to, draw',
}


def _classify_spot(obs, facing_bet):
    """Classify a postflop decision into a spot category."""
    made_hand = float(obs[IDX_MADE_HAND]) if len(obs) > IDX_MADE_HAND else 0.0
    has_flush_draw = float(obs[20]) > 0.5 if len(obs) > 20 else False
    has_straight_draw = float(obs[21]) > 0.5 if len(obs) > 21 else False

    prefix = 'facing_bet' if facing_bet else 'checked_to'

    if has_flush_draw or has_straight_draw:
        if made_hand < 0.125 - 0.01:
            return f'{prefix}_draw'

    if made_hand >= 0.25 - 0.01:
        return f'{prefix}_strong'
    elif made_hand >= 0.125 - 0.01:
        return f'{prefix}_pair'
    else:
        return f'{prefix}_air'


def _action_category(action_id):
    """Broad action categories for agreement analysis."""
    if action_id == FOLD:
        return 'passive'
    elif action_id == CALL:
        return 'flat'
    else:
        return 'aggressive'


def _get_tier(bb100):
    """Return tier label for a BB/100 win rate."""
    for lo, hi, label in TIERS:
        if lo <= bb100 < hi:
            return label
    return None


# ── Main analysis function ──────────────────────────────────────────────────

def analyze(
    db: AnalysisDB,
    model: ModelWrapper,
    good_players: Set[str],
    min_sample: int = 20,
) -> dict:
    """Run benchmark analysis comparing model vs winning players.

    Args:
        db: AnalysisDB with parsed decision points.
        model: ModelWrapper with loaded models.
        good_players: Set of known good player names.
        min_sample: Minimum DPs per tier to report.

    Returns:
        Dictionary with all analysis results.
    """
    # Compute player win rates from parsed data
    player_bb100 = compute_player_winrates(db)

    # Filter to good players in data
    good_in_data = {name for name in good_players if name in player_bb100}
    print(f'  Good players found in data: {len(good_in_data)} / {len(good_players)}')

    # Map good players to their hands where they showed cards
    good_hands_with_cards = set()
    good_player_cards = defaultdict(set)  # hand_id -> set of good player names with cards
    for hand in db.hands:
        for p in hand.active_players:
            if p.name in good_in_data and p.hole_cards:
                good_player_cards[hand.hand_id].add(p.name)

    # Counters
    tier_total = defaultdict(int)
    tier_agree = defaultdict(int)
    tier_agree_category = defaultdict(int)
    tier_disagree_pot_weighted = defaultdict(float)
    tier_total_pot = defaultdict(float)

    spot_total = defaultdict(lambda: defaultdict(int))
    spot_agree = defaultdict(lambda: defaultdict(int))
    spot_model_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    spot_player_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    street_total = defaultdict(lambda: defaultdict(int))
    street_agree = defaultdict(lambda: defaultdict(int))

    n_total = 0
    n_agree = 0
    n_agree_cat = 0
    n_facing_bet = 0
    n_checked_to = 0
    n_skipped_preflop = 0
    n_skipped_no_cards = 0

    for dp in db.decision_points:
        # Only analyze good players with shown cards
        if dp.hand_id not in good_player_cards:
            continue
        if dp.player_name not in good_player_cards[dp.hand_id]:
            continue

        # Skip preflop
        if dp.street == 'preflop':
            n_skipped_preflop += 1
            continue

        # Skip if no hole cards
        if dp.hole_cards is None:
            n_skipped_no_cards += 1
            continue

        obs = dp.observation
        legal = dp.legal_actions
        actual = dp.actual_action
        pot_dollars = dp.pot_dollars

        facing_bet = model.is_facing_bet(obs)

        if facing_bet:
            n_facing_bet += 1
        else:
            n_checked_to += 1

        # Get model action
        try:
            m_action, m_strategy = model.get_action(obs, legal)
        except Exception:
            continue

        # Get player's tier
        tier = _get_tier(player_bb100.get(dp.player_name, 0.0))
        if tier is None:
            tier = TIERS[0][2]

        spot = _classify_spot(obs, facing_bet)

        exact_agree = (m_action == actual)
        cat_agree = (_action_category(m_action) == _action_category(actual))

        n_total += 1
        tier_total[tier] += 1
        spot_total[tier][spot] += 1
        street_total[tier][dp.street] += 1

        if exact_agree:
            n_agree += 1
            tier_agree[tier] += 1
            spot_agree[tier][spot] += 1
            street_agree[tier][dp.street] += 1

        if cat_agree:
            n_agree_cat += 1
            tier_agree_category[tier] += 1

        if not exact_agree:
            tier_disagree_pot_weighted[tier] += pot_dollars
        tier_total_pot[tier] += pot_dollars

        spot_model_actions[tier][spot][m_action] += 1
        spot_player_actions[tier][spot][actual] += 1

    # Aggregate spots across tiers
    agg_spot_total = defaultdict(int)
    agg_spot_agree = defaultdict(int)
    agg_spot_model = defaultdict(lambda: defaultdict(int))
    agg_spot_player = defaultdict(lambda: defaultdict(int))

    for tier in spot_total:
        for spot in spot_total[tier]:
            agg_spot_total[spot] += spot_total[tier][spot]
            agg_spot_agree[spot] += spot_agree[tier][spot]
            for a in spot_model_actions[tier][spot]:
                agg_spot_model[spot][a] += spot_model_actions[tier][spot][a]
            for a in spot_player_actions[tier][spot]:
                agg_spot_player[spot][a] += spot_player_actions[tier][spot][a]

    return {
        'n_total': n_total,
        'n_agree': n_agree,
        'n_agree_cat': n_agree_cat,
        'n_facing_bet': n_facing_bet,
        'n_checked_to': n_checked_to,
        'n_skipped_preflop': n_skipped_preflop,
        'n_skipped_no_cards': n_skipped_no_cards,
        'tier_total': dict(tier_total),
        'tier_agree': dict(tier_agree),
        'tier_agree_category': dict(tier_agree_category),
        'tier_disagree_pot_weighted': dict(tier_disagree_pot_weighted),
        'tier_total_pot': dict(tier_total_pot),
        'spot_total': {t: dict(v) for t, v in spot_total.items()},
        'spot_agree': {t: dict(v) for t, v in spot_agree.items()},
        'agg_spot_total': dict(agg_spot_total),
        'agg_spot_agree': dict(agg_spot_agree),
        'agg_spot_model': {s: dict(v) for s, v in agg_spot_model.items()},
        'agg_spot_player': {s: dict(v) for s, v in agg_spot_player.items()},
        'street_total': {t: dict(v) for t, v in street_total.items()},
        'street_agree': {t: dict(v) for t, v in street_agree.items()},
        'player_bb100': player_bb100,
        'good_in_data': good_in_data,
        'min_sample': min_sample,
        'total_hands': db.total_showdown_hands,
    }


# ── Report printing ─────────────────────────────────────────────────────────

def print_report(results: dict):
    """Print the benchmark analysis report."""
    n_total = results['n_total']
    n_agree = results['n_agree']
    n_agree_cat = results['n_agree_cat']
    min_sample = results.get('min_sample', 20)

    print('\n' + '=' * 78)
    print('BENCHMARK: MODEL vs WINNING PLAYERS')
    print('=' * 78)

    print(f'Hands parsed: {results["total_hands"]}, '
          f'Decision points: {n_total} '
          f'(facing bet: {results["n_facing_bet"]}, '
          f'checked to: {results["n_checked_to"]})')
    print(f'Preflop skipped: {results["n_skipped_preflop"]}, '
          f'No cards skipped: {results["n_skipped_no_cards"]}')

    # Overall agreement
    print(f'\n--- Overall Agreement ---')
    if n_total > 0:
        print(f'  Exact action match:  {n_agree:>6}/{n_total:<6} = {100*n_agree/n_total:.1f}%')
        print(f'  Category match:      {n_agree_cat:>6}/{n_total:<6} = {100*n_agree_cat/n_total:.1f}%')
        print(f'    (Category: passive=fold, flat=call/check, aggressive=raise/allin)')
    else:
        print('  No decision points found.')
        return

    # Per-tier agreement
    print(f'\n--- Agreement by Win-Rate Tier ---')
    print(f'{"Tier":<16} {"DPs":>6} {"Exact":>7} {"Category":>9} {"Disagree$/pot$":>14}')
    print('-' * 58)
    tier_total = results['tier_total']
    tier_agree = results['tier_agree']
    tier_agree_cat = results['tier_agree_category']
    tier_dpw = results['tier_disagree_pot_weighted']
    tier_tpot = results['tier_total_pot']

    for _, _, label in TIERS:
        t = tier_total.get(label, 0)
        if t < min_sample:
            print(f'{label:<16} {t:>6}   (insufficient sample)')
            continue
        a = tier_agree.get(label, 0)
        ac = tier_agree_cat.get(label, 0)
        dpw = tier_dpw.get(label, 0)
        tpot = tier_tpot.get(label, 0)
        dpw_pct = (dpw / tpot * 100) if tpot > 0 else 0
        print(f'{label:<16} {t:>6} {100*a/t:>6.1f}% {100*ac/t:>8.1f}% {dpw_pct:>13.1f}%')

    # Per-street agreement
    print(f'\n--- Agreement by Street ---')
    print(f'{"Street":<10} ', end='')
    for _, _, label in TIERS:
        print(f'  {label:<16}', end='')
    print()
    print('-' * 62)
    s_total = results['street_total']
    s_agree = results['street_agree']
    for street in ['flop', 'turn', 'river']:
        print(f'{street:<10} ', end='')
        for _, _, label in TIERS:
            t = s_total.get(label, {}).get(street, 0)
            a = s_agree.get(label, {}).get(street, 0)
            if t > 0:
                print(f'  {100*a/t:>5.1f}% ({t:>5})', end='')
            else:
                print(f'       -       ', end='')
        print()

    # Per-spot breakdown
    agg_spot_total = results['agg_spot_total']
    agg_spot_agree = results['agg_spot_agree']
    agg_spot_model = results['agg_spot_model']
    agg_spot_player = results['agg_spot_player']

    print(f'\n--- Agreement by Spot Type ---')
    print(f'{"Spot":<32} {"DPs":>6} {"Agree":>7}  {"Model top action":>16}  {"Player top action":>17}')
    print('-' * 88)

    spot_order = ['facing_bet_air', 'facing_bet_draw', 'facing_bet_pair',
                  'facing_bet_strong', 'checked_to_air', 'checked_to_draw',
                  'checked_to_pair', 'checked_to_strong']

    for spot_key in spot_order:
        label = SPOT_NAMES.get(spot_key, spot_key)
        t = agg_spot_total.get(spot_key, 0)
        if t == 0:
            continue
        a = agg_spot_agree.get(spot_key, 0)

        m_acts = agg_spot_model.get(spot_key, {})
        m_top = max(m_acts, key=m_acts.get) if m_acts else -1
        m_pct = (m_acts[m_top] / t * 100) if m_top >= 0 else 0

        p_acts = agg_spot_player.get(spot_key, {})
        p_top = max(p_acts, key=p_acts.get) if p_acts else -1
        p_pct = (p_acts[p_top] / t * 100) if p_top >= 0 else 0

        m_name = ACTION_NAMES[m_top] if 0 <= m_top < len(ACTION_NAMES) else '?'
        p_name = ACTION_NAMES[p_top] if 0 <= p_top < len(ACTION_NAMES) else '?'

        print(f'{label:<32} {t:>6} {100*a/t:>6.1f}%  '
              f'{m_name:>8} ({m_pct:>4.1f}%)  '
              f'{p_name:>9} ({p_pct:>4.1f}%)')

    # Action distribution comparison
    print(f'\n--- Action Distribution: Model vs Winners ---')
    print(f'{"":>32}  {"Fold":>7} {"Call":>7} {"Rse1/2":>7} {"RsePot":>7} {"AllIn":>7}')
    print('-' * 82)

    for spot_key in spot_order:
        label = SPOT_NAMES.get(spot_key, spot_key)
        t = agg_spot_total.get(spot_key, 0)
        if t == 0:
            continue

        m_acts = agg_spot_model.get(spot_key, {})
        p_acts = agg_spot_player.get(spot_key, {})

        m_str = ''
        for a in range(5):
            pct = (m_acts.get(a, 0) / t * 100) if t > 0 else 0
            m_str += f' {pct:>6.1f}%'
        print(f'{label + " (model)":<32} {m_str}')

        p_str = ''
        for a in range(5):
            pct = (p_acts.get(a, 0) / t * 100) if t > 0 else 0
            p_str += f' {pct:>6.1f}%'
        print(f'{"  (winners)":<32} {p_str}')

    # Estimated model strength
    print(f'\n--- Estimated Model Strength ---')
    tier_weights = {TIERS[0][2]: 1.0, TIERS[1][2]: 1.5, TIERS[2][2]: 2.0}
    weighted_agree = 0.0
    weighted_total = 0.0
    for _, _, label in TIERS:
        t = tier_total.get(label, 0)
        a = tier_agree.get(label, 0)
        w = tier_weights.get(label, 1.0)
        if t >= min_sample:
            weighted_agree += (a / t) * w
            weighted_total += w

    if weighted_total > 0:
        weighted_score = weighted_agree / weighted_total * 100
        print(f'  Weighted agreement score: {weighted_score:.1f}%')
        print(f'    (Weights: 5-10=1.0x, 10-15=1.5x, 15+=2.0x)')

    # Aggression alignment
    total_model_agg = 0
    total_player_agg = 0
    for spot_key in agg_spot_model:
        for a in [RAISE_HALF, RAISE_POT, ALL_IN]:
            total_model_agg += agg_spot_model[spot_key].get(a, 0)
            total_player_agg += agg_spot_player[spot_key].get(a, 0)

    if n_total > 0:
        model_agg_pct = total_model_agg / n_total * 100
        player_agg_pct = total_player_agg / n_total * 100
        diff = model_agg_pct - player_agg_pct
        direction = 'more aggressive' if diff > 0 else 'more passive'
        print(f'  Model aggression: {model_agg_pct:.1f}%, '
              f'Winners aggression: {player_agg_pct:.1f}% '
              f'(model is {abs(diff):.1f}% {direction})')

    total_model_fold = sum(agg_spot_model[s].get(FOLD, 0) for s in agg_spot_model)
    total_player_fold = sum(agg_spot_player[s].get(FOLD, 0) for s in agg_spot_player)
    if n_total > 0:
        model_fold_pct = total_model_fold / n_total * 100
        player_fold_pct = total_player_fold / n_total * 100
        print(f'  Model fold rate:  {model_fold_pct:.1f}%, '
              f'Winners fold rate:  {player_fold_pct:.1f}%')

    total_dpw = sum(tier_dpw.values())
    total_tpot = sum(tier_tpot.values())
    if total_tpot > 0:
        print(f'  Disagreement cost: {total_dpw:.2f} (pot-weighted) of '
              f'${total_tpot:.2f} total pot')
        print(f'    = {total_dpw/total_tpot*100:.1f}% of pot at risk in disagreements')

    print(f'\n{"=" * 78}')
