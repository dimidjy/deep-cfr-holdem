#!/usr/bin/env python3
"""Find and explain every spot where the model disagrees with your annotations.

Loads annotated spot JSONs from logs/action_debug/, runs the model on each,
and prints detailed context for spots where the model's argmax action differs
from `right_action`.

Useful for understanding WHERE the model is leaking and WHY.

Usage:
  # Show all disagreements (default)
  python3 scripts/inspect_spot_disagreements.py --model models/best/checkpoint_iter_2800.pt

  # Filter to specific annotated action (e.g. show only spots you flagged as Call but model folded)
  python3 scripts/inspect_spot_disagreements.py --model ... --right-action Call

  # Filter to specific model action (e.g. show only spots where model folded)
  python3 scripts/inspect_spot_disagreements.py --model ... --model-action Fold

  # Show only major disagreements (skip raise-size mismatches)
  python3 scripts/inspect_spot_disagreements.py --model ... --major-only

  # Sort by model confidence in wrong action (most confidently wrong first)
  python3 scripts/inspect_spot_disagreements.py --model ... --sort-by confidence

  # Limit output count
  python3 scripts/inspect_spot_disagreements.py --model ... --limit 20
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


_RAISE_ACTIONS = {2, 3, 4, 5}
_ACTION_NAMES = {
    0: 'Fold', 1: 'Check/Call', 2: 'Raise½', 3: 'RaisePot',
    4: 'AllIn', 5: 'RaiseMin',
}
_NAME_TO_ACTION = {v.lower(): k for k, v in _ACTION_NAMES.items()}
_NAME_TO_ACTION['call'] = 1
_NAME_TO_ACTION['check'] = 1

_VALID_OBS_SIZES = {50, 51, 53, 54, 55, 56}


def _load_spot(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if data.get('deleted'):
        return None
    if not isinstance(data.get('right_action'), int):
        return None
    if not isinstance(data.get('obs_vector'), list):
        return None
    if len(data['obs_vector']) not in _VALID_OBS_SIZES:
        return None
    data['_path'] = str(path)
    return data


def load_annotated_spots(spots_dir):
    path = Path(spots_dir)
    if not path.is_dir():
        return []
    spots = []
    for p in sorted(path.glob('*.json')):
        data = _load_spot(p)
        if data is not None:
            spots.append(data)
    return spots


def score_action(model_action, right_action):
    if model_action == right_action:
        return 1.0
    if model_action in _RAISE_ACTIONS and right_action in _RAISE_ACTIONS:
        return 0.5
    return 0.0


def format_cards(cards):
    """Format list of {rank, suit} dicts as 'A♥ 2♥'."""
    suit_map = {'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'}
    return ' '.join(f"{c['rank']}{suit_map.get(c['suit'], c['suit'])}" for c in cards)


def format_position(my_seat, button_seat, num_active):
    rel = (my_seat - button_seat) % 6
    pos_map = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}
    return pos_map.get(rel, f'rel{rel}')


def format_strategy(strat):
    """Show non-zero actions with probabilities."""
    parts = []
    for i, p in enumerate(strat):
        if p > 0.01:
            parts.append(f'{_ACTION_NAMES.get(i, str(i))}={p*100:.0f}%')
    return ' '.join(parts) if parts else '(empty)'


def render_spot(spot, strategy, model_action, right_action, score):
    """Render a single disagreement with all relevant context."""
    gs = spot.get('game_state', {})
    feats = spot.get('features', {})
    fname = Path(spot['_path']).name

    lines = []
    lines.append(f"━━━ {fname} ━━━")

    # Hand info
    hero_seat = gs.get('my_seat', 0)
    players = gs.get('players', [])
    hole = []
    for p in players:
        if p.get('seat') == hero_seat:
            hole = p.get('hole_cards', [])
            break
    board = gs.get('community_cards', [])
    btn = gs.get('button_seat', 0)
    n_active = gs.get('num_active_players', 0)
    pos = format_position(hero_seat, btn, n_active) if players else '?'

    if hole:
        lines.append(f"  Hand: {format_cards(hole):<14} Board: {format_cards(board)}")
    else:
        lines.append(f"  (no hole cards)  Board: {format_cards(board)}")
    lines.append(f"  Street: {gs.get('street', '?'):<7} Pos: {pos:<5} "
                 f"Active: {n_active}-way  Pot: {gs.get('pot_total', 0):.1f} BB")

    # Decision context
    call_amt = gs.get('call_amount', 0)
    pot_odds = feats.get('pot_odds', 0)
    equity = feats.get('hand_strength', 0)
    facing_bet = feats.get('facing_bet', 0)
    if facing_bet:
        bet_to_pot = feats.get('bet_to_pot_norm', 0)
        lines.append(f"  Facing bet: {call_amt:.1f} BB (bet/pot={bet_to_pot:.2f})  "
                     f"PotOdds={pot_odds:.2f}  Equity={equity:.2f}")
    else:
        lines.append(f"  No bet to face         PotOdds=0.00      Equity={equity:.2f}")

    # Hand-class flags
    flags = []
    if feats.get('hero_flush_participation', 0) >= 0.99:
        flags.append('NFD' if hole and hole[0]['rank'] == 'A' else 'flush-draw')
    elif feats.get('hero_flush_participation', 0) >= 0.5:
        flags.append('backdoor-flush')
    if feats.get('hero_straight_participation', 0) >= 0.49:
        flags.append('straight-draw')
    made = feats.get('made_hand_cat', 0)
    if made >= 0.625:
        flags.append('flush+')
    elif made >= 0.5:
        flags.append('straight')
    elif made >= 0.375:
        flags.append('trips')
    elif made >= 0.25:
        flags.append('two-pair')
    elif made >= 0.125:
        flags.append('one-pair')
    if feats.get('top_pair', 0) > 0.5:
        flags.append('top-pair')
    if feats.get('board_straight_texture', 0) > 0.7:
        flags.append('wet-board')
    if flags:
        lines.append(f"  Hand class: {', '.join(flags)}")

    # The disagreement
    right_name = _ACTION_NAMES.get(right_action, str(right_action))
    model_name = _ACTION_NAMES.get(model_action, str(model_action))
    score_str = 'PARTIAL' if score == 0.5 else 'MISS'
    lines.append(f"  → Right action: {right_name:<11}   Model: {model_name:<11}   [{score_str}]")
    lines.append(f"  Model strategy: {format_strategy(strategy)}")

    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', required=True, help='Path to DeepCFR .pt checkpoint')
    ap.add_argument('--spots-dir', default='logs/action_debug',
                    help='Directory of spot JSONs (default: logs/action_debug)')
    ap.add_argument('--right-action', default='',
                    help='Filter to spots with this annotated action '
                         '(Fold/Call/Check/Raise½/RaisePot/AllIn/RaiseMin)')
    ap.add_argument('--model-action', default='',
                    help='Filter to spots where model picked this action')
    ap.add_argument('--major-only', action='store_true',
                    help='Skip partial-credit raise-size mismatches')
    ap.add_argument('--sort-by', choices=['file', 'confidence', 'category'],
                    default='confidence',
                    help='Sort order (default: confidence — most confidently '
                         'wrong first)')
    ap.add_argument('--limit', type=int, default=0,
                    help='Show only first N disagreements (0 = all)')
    ap.add_argument('--summary-only', action='store_true',
                    help='Skip per-spot output, only show summary tables')
    args = ap.parse_args()

    # Parse filter args
    filter_right = None
    if args.right_action:
        filter_right = _NAME_TO_ACTION.get(args.right_action.lower())
        if filter_right is None:
            print(f'Unknown right action: {args.right_action}')
            sys.exit(1)
    filter_model = None
    if args.model_action:
        filter_model = _NAME_TO_ACTION.get(args.model_action.lower())
        if filter_model is None:
            print(f'Unknown model action: {args.model_action}')
            sys.exit(1)

    # Load model (using validate_spot_logs path for compatibility)
    from src.training import DeepCFR
    from src.utils import load_config
    from src.engine.factory import make_engine
    config_path = Path(__file__).resolve().parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))
    engine = make_engine(num_players=config['game']['num_players'])
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()
    deep_cfr = DeepCFR(
        observation_size=obs_size, action_size=act_size,
        num_players=config['game']['num_players'],
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size'],
        dropout=config['network'].get('dropout', 0.1),
        device='cpu',
    )
    deep_cfr.load_models(args.model)

    spots = load_annotated_spots(args.spots_dir)
    print(f'Loaded {len(spots)} annotated spots from {args.spots_dir}')

    # Run model on each, collect disagreements
    disagreements = []
    summary_by_pair = defaultdict(int)  # (right, model) -> count
    summary_by_right = Counter()
    n_exact = n_partial = n_miss = 0
    model_obs_size = getattr(deep_cfr, 'observation_size', None)

    for spot in spots:
        right = int(spot['right_action'])
        obs = np.asarray(spot['obs_vector'], dtype=np.float32)
        if model_obs_size and len(obs) > model_obs_size:
            obs = obs[:model_obs_size]
        elif model_obs_size and len(obs) < model_obs_size:
            padded = np.zeros(model_obs_size, dtype=np.float32)
            padded[:len(obs)] = obs
            obs = padded

        gs = spot.get('game_state', {})
        legal = gs.get('legal_actions') or list(range(6))
        player_idx = gs.get('my_seat', 0) % config['game']['num_players']

        try:
            strategy = deep_cfr.get_strategy(obs, player_idx, legal)
        except Exception:
            continue

        model_action = int(np.argmax(strategy))
        score = score_action(model_action, right)

        if score == 1.0:
            n_exact += 1
            continue
        if score == 0.5:
            n_partial += 1
        else:
            n_miss += 1

        summary_by_pair[(right, model_action)] += 1
        summary_by_right[right] += 1

        if args.major_only and score == 0.5:
            continue
        if filter_right is not None and right != filter_right:
            continue
        if filter_model is not None and model_action != filter_model:
            continue

        confidence = float(strategy[model_action])
        disagreements.append({
            'spot': spot,
            'strategy': strategy,
            'model_action': model_action,
            'right_action': right,
            'score': score,
            'confidence': confidence,
        })

    # Sort
    if args.sort_by == 'confidence':
        disagreements.sort(key=lambda d: -d['confidence'])
    elif args.sort_by == 'category':
        disagreements.sort(key=lambda d: (d['right_action'], d['model_action'],
                                          -d['confidence']))
    # 'file' is default sorted order

    # Apply limit
    if args.limit > 0:
        rendered = disagreements[:args.limit]
    else:
        rendered = disagreements

    # Print
    print()
    print('═' * 70)
    print(f'  DISAGREEMENTS  (showing {len(rendered)} of {len(disagreements)} filtered, '
          f'{n_miss + n_partial} total non-exact / {len(spots)} spots)')
    print('═' * 70)

    if not args.summary_only:
        for d in rendered:
            print(render_spot(d['spot'], d['strategy'], d['model_action'],
                             d['right_action'], d['score']))

    # Summary table
    print('═' * 70)
    print('  SUMMARY')
    print('═' * 70)
    total_scored = n_exact + n_partial + n_miss
    if total_scored > 0:
        score_total = n_exact + 0.5 * n_partial
        print(f'  Total scored: {total_scored}')
        print(f'    Exact match:   {n_exact:>5}  ({100 * n_exact / total_scored:.1f}%)')
        print(f'    Partial (raise size): {n_partial:>5}  ({100 * n_partial / total_scored:.1f}%)')
        print(f'    Major miss:    {n_miss:>5}  ({100 * n_miss / total_scored:.1f}%)')
        print(f'    Score: {score_total:.1f}/{total_scored} ({100 * score_total / total_scored:.1f}%)')

    if summary_by_pair:
        print()
        print('  Disagreement matrix (right_action → model_action):')
        print(f'    {"Right":<12} → {"Model":<12} {"Count":>6}')
        print('    ' + '-' * 36)
        for (r, m), c in sorted(summary_by_pair.items(), key=lambda x: -x[1]):
            r_name = _ACTION_NAMES.get(r, str(r))
            m_name = _ACTION_NAMES.get(m, str(m))
            print(f'    {r_name:<12} → {m_name:<12} {c:>6}')


if __name__ == '__main__':
    main()
