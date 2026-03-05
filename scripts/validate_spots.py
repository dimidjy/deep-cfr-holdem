"""
Poker spot validator — tests model strategy on canonical spots across all streets.

Runs N random games and at every decision point classifies the situation into
a well-defined category, then records the model's action probabilities.
Results are compared against correctness thresholds.

Usage:
  python3 scripts/validate_spots.py --model models/final_model_enhanced.pt
  python3 scripts/validate_spots.py --model models/checkpoint_iter_500.pt --samples 2000
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import warnings
import logging
import numpy as np
from collections import defaultdict
from pathlib import Path

import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config

# ---------------------------------------------------------------------------
# Action indices (texas_holdem_no_limit_v6)
# ---------------------------------------------------------------------------
FOLD       = 0
CALL       = 1   # also covers check
RAISE_HALF = 2
RAISE_FULL = 3
RAISE_2X   = 4
ALL_IN     = 5
AGGRESSIVE = [RAISE_HALF, RAISE_FULL, RAISE_2X, ALL_IN]

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
            'T':10,'J':11,'Q':12,'K':13,'A':14}


def _rank(ch: str) -> int:
    return RANK_MAP.get(ch, 0)


# ---------------------------------------------------------------------------
# Preflop hand classifier
# ---------------------------------------------------------------------------

def classify_preflop(cards):
    """
    Classify 2 hole cards into a preflop validation category.
    Returns None for hands we don't track.
    cards: list of strings like ['As', '2h']
    """
    if len(cards) < 2:
        return None
    r1, r2 = _rank(cards[0][0]), _rank(cards[1][0])
    s1, s2 = cards[0][1], cards[1][1]
    hi, lo    = max(r1, r2), min(r1, r2)
    is_pair   = r1 == r2
    is_suited = s1 == s2

    if is_pair:
        if hi == 14: return 'AA'
        if hi == 13: return 'KK'
        if hi == 12: return 'QQ'
        if hi >= 10: return 'JJ-TT'
        return None

    if hi == 14 and lo == 13:
        return 'AKs' if is_suited else 'AKo'
    if hi == 14 and lo >= 11:
        return 'AQs-AJs' if is_suited else 'AQo-AJo'

    # Worst garbage: high ≤ 8, gap ≥ 3, offsuit
    if hi <= 8 and (hi - lo) >= 3 and not is_suited:
        if hi == 7 and lo == 2: return '72o'
        return 'trash_offsuit'

    return None


# ---------------------------------------------------------------------------
# Post-flop hand classifier
# ---------------------------------------------------------------------------

def classify_postflop(hole_cards, community_cards):
    """
    Classify the current made hand (flop/turn/river).
    Priority: monster > flush > set > two_pair > air.
    Returns None for unclassified / borderline hands.
    hole_cards, community_cards: list of strings like 'As', 'Kh', ...
    """
    if len(hole_cards) < 2 or len(community_cards) < 3:
        return None

    r1, r2 = _rank(hole_cards[0][0]), _rank(hole_cards[1][0])
    s1, s2 = hole_cards[0][1], hole_cards[1][1]
    board_ranks = [_rank(c[0]) for c in community_cards]
    board_suits  = [c[1]        for c in community_cards]
    is_pair   = r1 == r2
    is_suited = s1 == s2

    # --- MONSTER: full house or quads ---
    if is_pair:
        board_count = board_ranks.count(r1)
        if board_count >= 2:
            return 'monster'  # quads
        if board_count == 1:
            other = [r for r in board_ranks if r != r1]
            if len(other) > len(set(other)):
                return 'monster'  # set + board pair = full house
            return 'set'

    # --- FLUSH: both suited, 3+ board cards of same suit ---
    if is_suited and board_suits.count(s1) >= 3:
        return 'flush'

    # --- TWO PAIR: both hole cards pair different board cards ---
    if r1 != r2 and r1 in board_ranks and r2 in board_ranks:
        return 'two_pair'

    # --- COMPLETE AIR: no pair, no draw, weak high card ---
    # Conditions: neither hole card pairs board, not a pocket pair,
    # no flush draw (< 2 board cards of each hole card's suit), high card ≤ 9
    if (r1 not in board_ranks and r2 not in board_ranks
            and not is_pair and max(r1, r2) <= 9
            and board_suits.count(s1) < 2
            and board_suits.count(s2) < 2):
        return 'air'

    return None


# ---------------------------------------------------------------------------
# Spot definitions — grouped by street
# Each spot has:
#   category – matches classify_*() output
#   street   – 'preflop' | 'postflop' | 'any'
#   label    – display name
#   hint     – threshold description shown on FAIL
#   pass_fn  – lambda(avg_strategy, legal_action_set) → bool
# ---------------------------------------------------------------------------
SPOTS = [
    # ── Preflop: premium hands (should raise, almost never fold) ───────────
    {
        'category': 'AA',
        'street':   'preflop',
        'label':    'AA preflop',
        'hint':     'fold<10%  aggr>60%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.10 and _aggr(s, L) > 0.60,
    },
    {
        'category': 'KK',
        'street':   'preflop',
        'label':    'KK preflop',
        'hint':     'fold<10%  aggr>50%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.10 and _aggr(s, L) > 0.50,
    },
    {
        'category': 'QQ',
        'street':   'preflop',
        'label':    'QQ preflop',
        'hint':     'fold<15%  aggr>35%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.15 and _aggr(s, L) > 0.35,
    },
    {
        'category': 'JJ-TT',
        'street':   'preflop',
        'label':    'JJ-TT preflop',
        'hint':     'fold<20%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.20,
    },
    {
        'category': 'AKo',
        'street':   'preflop',
        'label':    'AKo preflop',
        'hint':     'fold<15%  aggr>40%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.15 and _aggr(s, L) > 0.40,
    },
    {
        'category': 'AKs',
        'street':   'preflop',
        'label':    'AKs preflop',
        'hint':     'fold<10%  aggr>50%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.10 and _aggr(s, L) > 0.50,
    },
    {
        'category': 'AQo-AJo',
        'street':   'preflop',
        'label':    'AQo-AJo preflop',
        'hint':     'fold<20%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.20,
    },
    # ── Preflop: trash (should fold often) ─────────────────────────────────
    {
        'category': '72o',
        'street':   'preflop',
        'label':    '7-2 offsuit (worst hand)',
        'hint':     'fold>30%',
        'pass_fn':  lambda s, L: s[FOLD] > 0.30,
    },
    {
        'category': 'trash_offsuit',
        'street':   'preflop',
        'label':    'Trash offsuit (hi≤8, gap≥3)',
        'hint':     'fold>25%',
        'pass_fn':  lambda s, L: s[FOLD] > 0.25,
    },
    # ── Post-flop: monster hands — virtually never fold ────────────────────
    {
        'category': 'monster',
        'street':   'postflop',
        'label':    'Full house / quads (any street)',
        'hint':     'fold<3%  aggr>35%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.03 and _aggr(s, L) > 0.35,
    },
    {
        'category': 'set',
        'street':   'postflop',
        'label':    'Set on board (any street)',
        'hint':     'fold<5%  aggr>30%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.05 and _aggr(s, L) > 0.30,
    },
    {
        'category': 'flush',
        'street':   'postflop',
        'label':    'Made flush (any street)',
        'hint':     'fold<5%  aggr>25%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.05 and _aggr(s, L) > 0.25,
    },
    {
        'category': 'two_pair',
        'street':   'postflop',
        'label':    'Two pair — both hole cards hit board',
        'hint':     'fold<10%  aggr>20%',
        'pass_fn':  lambda s, L: s[FOLD] < 0.10 and _aggr(s, L) > 0.20,
    },
    # ── Post-flop: air — don't bet/raise with nothing ──────────────────────
    {
        'category': 'air',
        'street':   'postflop',
        'label':    'Complete air (no pair/draw, hi≤9)',
        'hint':     'aggr<35%',
        'pass_fn':  lambda s, L: _aggr(s, L) < 0.35,
    },
]

# Helper used inside lambdas (lambdas can't use local def)
def _aggr(s, L):
    return sum(float(s[a]) for a in AGGRESSIVE if a in L)


# ---------------------------------------------------------------------------
# Core collection loop
# ---------------------------------------------------------------------------

def collect_samples(deep_cfr, engine, num_games: int = 1000):
    """
    Run num_games hands and record model strategy at every decision point,
    bucketed by situation category.
    Returns dict[category → list of (strategy_array, legal_set)]
    """
    samples = defaultdict(list)

    for _ in range(num_games):
        engine.reset()

        while not engine.is_done():
            agent     = engine.current_agent
            agent_idx = engine.agents.index(agent)
            stage     = engine.get_stage()
            obs       = engine.get_observation()
            legal     = engine.get_legal_actions()

            if not legal:
                break

            # Classify the current situation
            cards = engine.get_player_cards(agent)
            if stage == 'Preflop':
                cat = classify_preflop(cards)
            else:
                community = engine.get_community_cards()
                cat = classify_postflop(cards, community)

            # Record strategy before advancing (one get_strategy call, reused below)
            strategy = deep_cfr.get_strategy(obs, agent_idx, legal)
            if cat is not None:
                samples[cat].append((strategy.copy(), set(legal)))

            legal_probs = strategy[legal]
            legal_probs = legal_probs / legal_probs.sum()
            action      = int(np.random.choice(legal, p=legal_probs))
            engine.step(action)

    return samples


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_validation(deep_cfr, engine, num_games: int = 1000,
                   verbose: bool = True) -> dict:
    """
    Run validation and return a results dict.
    Prints a human-readable report if verbose=True.
    """
    for net in deep_cfr.advantage_networks + deep_cfr.strategy_networks:
        net.eval()

    samples = collect_samples(deep_cfr, engine, num_games)

    results = {}
    passed  = 0
    total   = 0

    if verbose:
        print()
        print('=' * 76)
        print(f'  SPOT VALIDATION  ({num_games} games sampled, '
              f'iter {deep_cfr.iteration})')
        print('=' * 76)

    current_street = None
    for spot in SPOTS:
        cat    = spot['category']
        street = spot['street']
        label  = spot['label']
        hint   = spot['hint']

        # Print section header when street changes
        if verbose and street != current_street:
            current_street = street
            section = {'preflop': '── PREFLOP ──', 'postflop': '── POST-FLOP ──'}.get(street, '')
            print(f'\n  {section}')
            print(f"  {'Spot':<34} {'N':>5}  {'fold%':>6}  {'call%':>6}  {'aggr%':>6}  Status")
            print('  ' + '-' * 70)

        if cat not in samples or not samples[cat]:
            if verbose:
                print(f"  {label:<34} {'—':>5}  (no samples)")
            results[cat] = {'n': 0, 'passed': None}
            continue

        cat_samples = samples[cat]
        n           = len(cat_samples)

        all_legal = set()
        for _, L in cat_samples:
            all_legal.update(L)

        avg_strategy = np.mean([s for s, _ in cat_samples], axis=0)

        fold_pct = float(avg_strategy[FOLD])
        call_pct = float(avg_strategy[CALL])
        aggr_pct = _aggr(avg_strategy, all_legal)

        passed_spot = spot['pass_fn'](avg_strategy, all_legal)

        # Mark low-sample results as uncertain
        low_n  = n < 20
        status = ('PASS ✓' if passed_spot else f'FAIL  (want: {hint})')
        if low_n:
            status += ' [low N]'

        results[cat] = {
            'n':        n,
            'fold_pct': fold_pct,
            'call_pct': call_pct,
            'aggr_pct': aggr_pct,
            'passed':   passed_spot,
            'low_n':    low_n,
        }

        # Don't count low-N postflop spots toward pass/fail totals
        if not low_n:
            if passed_spot:
                passed += 1
            total += 1

        if verbose:
            print(f"  {label:<34} {n:>5}  {fold_pct:>6.1%}  {call_pct:>6.1%}"
                  f"  {aggr_pct:>6.1%}  {status}")

    if verbose:
        print()
        print('  ' + '-' * 70)
        verdict = ('Model shows basic poker understanding'
                   if total > 0 and passed >= total * 0.6
                   else 'Model needs more training')
        print(f"  Result: {passed}/{total} spots passed (low-N excluded)  —  {verdict}")
        print('=' * 76)
        print()

    results['_summary'] = {'passed': passed, 'total': total}
    return results


def validation_summary_line(results: dict) -> str:
    """Compact one-liner for embedding in training log output."""
    parts = []
    for spot in SPOTS:
        cat = spot['category']
        if cat in results and results[cat].get('n', 0) >= 5:
            r   = results[cat]
            tag = 'ok' if r['passed'] else 'BAD'
            if r.get('low_n'):
                tag += '?'
            parts.append(
                f"{cat}:fold={r['fold_pct']:.0%}"
                f",aggr={r['aggr_pct']:.0%}({tag})"
            )
    s = results.get('_summary', {})
    return f"[spots {s.get('passed',0)}/{s.get('total',0)}] " + '  '.join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Validate model strategy on canonical spots across all streets'
    )
    parser.add_argument('--model',   type=str, required=True,
                        help='Path to .pt checkpoint file')
    parser.add_argument('--samples', type=int, default=1000,
                        help='Number of games to sample (default: 1000)')
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f'Model not found: {model_path}')
        sys.exit(1)

    # Auto-detect player count from first-layer weight shape
    num_players = 6
    try:
        ckpt       = torch.load(str(model_path), map_location='cpu')
        input_size = ckpt['advantage_networks'][0]['network.0.weight'].shape[1]
        for p in range(2, 10):
            test_eng = EnhancedPokerEngine(
                num_players=p,
                add_position_features=True,
                add_pot_odds_features=True,
                add_hand_strength_features=True,
                add_stack_features=True,
            )
            if test_eng.get_observation_space_size() == input_size:
                num_players = p
                print(f'Auto-detected: {num_players} players (obs_size={input_size})')
                break
    except Exception as e:
        print(f'Could not auto-detect ({e}), defaulting to {num_players} players')

    engine = EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
    )

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config      = load_config(str(config_path))

    deep_cfr = DeepCFR(
        observation_size=engine.get_observation_space_size(),
        action_size=engine.get_action_space_size(),
        num_players=num_players,
        hidden_layers=config['network']['hidden_layers'],
        device='cpu',
    )
    deep_cfr.load_models(str(model_path))
    print(f'Loaded: {model_path.name}  (iteration {deep_cfr.iteration})')

    run_validation(deep_cfr, engine, num_games=args.samples)


if __name__ == '__main__':
    main()
