"""
Benchmark a poker model against known winning players from hand history data.

Parses hand histories, finds postflop showdown hands where good players acted,
computes what the model would do at each decision point, and reports agreement
rates grouped by winner win-rate tier and spot type.

Usage:
  # Single model:
  python3 scripts/analysis/benchmark_vs_winners.py --model models/best/cql_6p_v7.pt --max-hands 20000

  # Split models (facing_bet + checked_to):
  python3 scripts/analysis/benchmark_vs_winners.py \
      --model-facing-bet models/best/cql_6p_facing_bet_v7.pt \
      --model-checked-to models/best/cql_6p_checked_to_v3.pt \
      --max-hands 20000
"""

import sys
import os

# Project root = parent of scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

import argparse
import multiprocessing as mp
import time
import warnings
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder
from scripts.train_cql import QNetwork, CQLModelWrapper

# Action indices (engine 5-action space)
FOLD = 0
CALL = 1
RAISE_HALF = 2
RAISE_POT = 3
ALL_IN = 4

ACTION_NAMES = ['Fold', 'Call', 'Raise1/2', 'RaisePot', 'AllIn']

# Observation indices
_MADE_HAND_IDX = 19
_POT_ODDS_IDX = 10  # call_norm — obs[10] > 0.001 means facing a bet

# Win rate tier boundaries (BB/100)
TIERS = [
    (5.0, 10.0, '5-10 BB/100'),
    (10.0, 15.0, '10-15 BB/100'),
    (15.0, 999.0, '15+ BB/100'),
]

# Spot classification for postflop
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
    made_hand = float(obs[_MADE_HAND_IDX]) if len(obs) > _MADE_HAND_IDX else 0.0
    has_flush_draw = float(obs[20]) > 0.5 if len(obs) > 20 else False
    has_straight_draw = float(obs[21]) > 0.5 if len(obs) > 21 else False

    prefix = 'facing_bet' if facing_bet else 'checked_to'

    if has_flush_draw or has_straight_draw:
        if made_hand < 0.125 - 0.01:  # draw without made hand
            return f'{prefix}_draw'

    if made_hand >= 0.25 - 0.01:  # two pair+
        return f'{prefix}_strong'
    elif made_hand >= 0.125 - 0.01:  # one pair
        return f'{prefix}_pair'
    else:
        return f'{prefix}_air'


def _action_category(action_id):
    """Broad action categories for agreement analysis."""
    if action_id == FOLD:
        return 'passive'  # fold = giving up
    elif action_id == CALL:
        return 'flat'     # call/check = flat
    else:
        return 'aggressive'  # any raise/all-in


def load_model(path, device='cpu'):
    """Load a CQL model from checkpoint, return (wrapper, obs_size, action_size)."""
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get('model_type') not in ('cql', 'iql'):
        raise ValueError(f"Expected CQL/IQL model, got: {checkpoint.get('model_type')}")

    obs_size = checkpoint['obs_size']
    action_size = checkpoint['action_size']
    hidden = checkpoint.get('hidden_layers', [512, 512, 256])
    dropout = checkpoint.get('dropout', 0.1)

    q_net = QNetwork(obs_size, action_size, hidden, dropout)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    q_net.eval()

    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size,
        num_players=6, device=device,
        iteration=checkpoint.get('epoch', 0),
        temperature=checkpoint.get('temperature', 1.0),
        exclude_fold=False,
    )

    has_bh = checkpoint.get('behavioral_stats', False) or (obs_size - 41) >= 8
    has_evo = (obs_size - 41) >= 9
    return wrapper, obs_size, action_size, has_bh, has_evo


def _compute_player_net(hand):
    """Compute net profit/loss for each player in a hand (in dollars)."""
    player_put_in = defaultdict(float)
    round_committed = defaultdict(float)
    current_street = 'preflop'

    for action in hand.actions:
        if action.street != current_street:
            current_street = action.street
            round_committed.clear()

        name = action.player_name
        if action.is_blind:
            player_put_in[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'call':
            player_put_in[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'bet':
            player_put_in[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'raise':
            to_amt = action.amount_dollars
            already = round_committed.get(name, 0)
            delta = to_amt - already
            if delta > 0:
                player_put_in[name] += delta
            round_committed[name] = to_amt

    player_won = defaultdict(float)
    for w in hand.result.winners:
        player_won[w['name']] += w['amount_dollars']

    player_net = {}
    for p in hand.active_players:
        player_net[p.name] = player_won.get(p.name, 0.0) - player_put_in.get(p.name, 0.0)
    return player_net


def _parse_worker(file_path):
    """Parse one HH file and return (hands, per_hand_player_data)."""
    try:
        hands = parse_file(str(file_path))
    except Exception:
        return []

    results = []
    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != 6:
            continue
        if not hand.has_showdown:
            continue

        bb = hand.big_blind_dollars
        active_names = set(p.name for p in hand.active_players)
        player_net = _compute_player_net(hand)

        results.append({
            'hand': hand,
            'bb': bb,
            'players': active_names,
            'player_net': player_net,
        })
    return results


def compute_player_stats(all_results):
    """Compute per-player stats: hands played, total BB won/lost."""
    player_hands = defaultdict(int)
    player_bb_net = defaultdict(float)

    for r in all_results:
        bb = r['bb']
        for name in r['players']:
            player_hands[name] += 1
            net_bb = r['player_net'].get(name, 0.0) / bb if bb > 0 else 0.0
            player_bb_net[name] += net_bb

    player_bb100 = {}
    for name in player_hands:
        n = player_hands[name]
        if n >= 100:
            player_bb100[name] = (player_bb_net[name] / n) * 100.0
    return player_hands, player_bb100


def get_tier(bb100):
    """Return tier label for a BB/100 win rate."""
    for lo, hi, label in TIERS:
        if lo <= bb100 < hi:
            return label
    return None


def model_action(model, obs, legal_actions, facing_bet, action_size, model_3bet=None):
    """Get model's preferred action and full strategy.

    For checked_to model: exclude fold from legal actions.
    """
    if not facing_bet:
        model_legal = [a for a in legal_actions if a != FOLD] or [CALL]
    else:
        model_legal = list(legal_actions)

    # Filter out action 5+ (RaiseMin not in model output)
    model_legal = [a for a in model_legal if a < 5] or [CALL]

    # Use 3bet model for large pots when facing bet
    active_model = model
    if facing_bet and model_3bet is not None:
        pot_bb = obs[8] * 600 if len(obs) > 8 else 0
        if pot_bb > 12:
            active_model = model_3bet

    strategy = active_model.get_strategy(obs, 0, model_legal)

    # Pick best action
    best_action = max(model_legal, key=lambda a: strategy[a])
    return best_action, strategy


def pad_observation(obs, target_size):
    """Pad observation to target size, computing equity_vs_odds if needed."""
    if len(obs) >= target_size:
        return obs[:target_size]
    padded = np.zeros(target_size, dtype=np.float32)
    padded[:len(obs)] = obs
    # Compute equity_vs_odds for the last dim if it's the +1 extra
    if len(obs) >= 19 and target_size > len(obs):
        pot_odds = obs[_POT_ODDS_IDX]
        equity = obs[18]  # hand_strength
        evo_idx = target_size - 1
        padded[evo_idx] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
    return padded


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark model vs winning players from hand histories')
    parser.add_argument('--model', type=str, default=None,
                        help='Single model path (used for both facing-bet and checked-to)')
    parser.add_argument('--model-facing-bet', type=str, default=None,
                        help='Model for facing-bet spots')
    parser.add_argument('--model-checked-to', type=str, default=None,
                        help='Model for checked-to spots')
    parser.add_argument('--model-facing-bet-3bet', type=str, default=None,
                        help='Model for facing-bet spots in 3bet pots (pot > 12BB)')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--good-players', type=str, default='data/good_players.txt',
                        help='Path to good players list')
    parser.add_argument('--max-hands', type=int, default=20000,
                        help='Max hands to process (0 = all)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parser workers (0 = auto)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for model inference')
    parser.add_argument('--min-sample', type=int, default=20,
                        help='Min decision points per tier to report')
    args = parser.parse_args()

    # Validate model args
    if args.model:
        fb_path = args.model
        ct_path = args.model
        split_models = False
    elif args.model_facing_bet and args.model_checked_to:
        fb_path = args.model_facing_bet
        ct_path = args.model_checked_to
        split_models = True
    else:
        parser.error('Provide --model or both --model-facing-bet and --model-checked-to')
        return

    # Load good players set
    good_players_path = os.path.join(_PROJECT_ROOT, args.good_players)
    good_players = set()
    with open(good_players_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                good_players.add(line)
    print(f'Good players loaded: {len(good_players)}')

    # Find HH files
    hh_dir = Path(os.path.join(_PROJECT_ROOT, args.hh_dir))
    hh_files = sorted(hh_dir.rglob('*.txt'))
    print(f'HH files found: {len(hh_files)}')

    # Parse FIRST (before loading torch/CUDA models — mp.Pool deadlocks after CUDA init)
    print(f'Parsing hand histories...')
    t0 = time.time()

    n_workers = args.workers if args.workers > 0 else min(mp.cpu_count(), 24)
    all_results = []
    total_hands = 0

    with mp.Pool(n_workers) as pool:
        for batch_results in pool.imap_unordered(_parse_worker, hh_files, chunksize=50):
            all_results.extend(batch_results)
            total_hands += len(batch_results)
            if (total_hands % 10000) < 100:
                print(f'  {total_hands} hands...', flush=True)
            if args.max_hands > 0 and total_hands >= args.max_hands:
                pool.terminate()
                break

    if args.max_hands > 0:
        all_results = all_results[:args.max_hands]

    parse_time = time.time() - t0
    print(f'Parsed {len(all_results)} showdown hands in {parse_time:.1f}s')

    # NOW load models (after mp.Pool is done)
    print(f'Loading models...')
    fb_model, fb_obs_size, fb_act_size, fb_bh, fb_evo = load_model(fb_path, args.device)
    if split_models:
        ct_model, ct_obs_size, ct_act_size, ct_bh, ct_evo = load_model(ct_path, args.device)
    else:
        ct_model, ct_obs_size, ct_act_size, ct_bh, ct_evo = (
            fb_model, fb_obs_size, fb_act_size, fb_bh, fb_evo)

    print(f'  Facing-bet model: obs={fb_obs_size}, actions={fb_act_size}, '
          f'behavioral={"ON" if fb_bh else "OFF"}, evo={"ON" if fb_evo else "OFF"}')
    if split_models:
        print(f'  Checked-to model: obs={ct_obs_size}, actions={ct_act_size}, '
              f'behavioral={"ON" if ct_bh else "OFF"}, evo={"ON" if ct_evo else "OFF"}')

    parse_time = time.time() - t0
    print(f'Parsed {len(all_results)} showdown hands in {parse_time:.1f}s')

    # Compute per-player win rates from parsed data
    print('Computing player win rates from parsed data...')
    player_hands, player_bb100 = compute_player_stats(all_results)

    # Filter to good players that appear in our data
    good_in_data = {name for name in good_players if name in player_bb100}
    print(f'Good players found in parsed data: {len(good_in_data)} / {len(good_players)}')

    # Show tier distribution
    tier_counts = defaultdict(int)
    for name in good_in_data:
        tier = get_tier(player_bb100[name])
        if tier:
            tier_counts[tier] += 1
    for _, _, label in TIERS:
        print(f'  {label}: {tier_counts.get(label, 0)} players')

    # Build observation builder (with behavioral stats if model needs it)
    need_bh = fb_bh or ct_bh
    obs_builder = ObservationBuilder(num_players=6)

    # Process decision points
    print(f'\nAnalyzing decision points...')
    t1 = time.time()

    # Counters per tier
    tier_total = defaultdict(int)
    tier_agree = defaultdict(int)
    tier_agree_category = defaultdict(int)  # agree on broad category
    tier_disagree_pot_weighted = defaultdict(float)
    tier_total_pot = defaultdict(float)

    # Counters per spot
    spot_total = defaultdict(lambda: defaultdict(int))
    spot_agree = defaultdict(lambda: defaultdict(int))
    spot_model_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    spot_player_actions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Per-street counters
    street_total = defaultdict(lambda: defaultdict(int))
    street_agree = defaultdict(lambda: defaultdict(int))

    # Overall counters
    n_total = 0
    n_agree = 0
    n_agree_cat = 0
    n_facing_bet = 0
    n_checked_to = 0
    n_skipped_preflop = 0
    n_skipped_no_cards = 0

    for r_idx, r in enumerate(all_results):
        hand = r['hand']
        bb = r['bb']

        # Check if any good player has shown cards in this hand
        good_with_cards = set()
        for p in hand.active_players:
            if p.name in good_in_data and p.hole_cards:
                good_with_cards.add(p.name)

        if not good_with_cards:
            continue

        # Build decision points
        try:
            dps = obs_builder.build_decision_points(
                hand,
                allow_unknown_cards=False,
                behavioral_stats=need_bh,
            )
        except Exception:
            continue

        for dp in dps:
            # Only analyze good players' decisions
            if dp.player_name not in good_with_cards:
                continue

            # Skip preflop (handled by ranges)
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

            # Determine facing bet: obs[10] > 0.001
            facing_bet = float(obs[_POT_ODDS_IDX]) > 0.001 if len(obs) > _POT_ODDS_IDX else False

            # Select model and pad observation
            if facing_bet:
                active_model = fb_model
                target_obs_size = fb_obs_size
                n_facing_bet += 1
            else:
                active_model = ct_model
                target_obs_size = ct_obs_size
                n_checked_to += 1

            padded_obs = pad_observation(obs, target_obs_size)

            # Get model action
            try:
                m_action, m_strategy = model_action(
                    active_model, padded_obs, legal, facing_bet,
                    fb_act_size if facing_bet else ct_act_size,
                    model_3bet=model_fb_3bet if facing_bet else None)
            except Exception:
                continue

            # Get player's tier
            player_name = dp.player_name
            tier = get_tier(player_bb100.get(player_name, 0.0))
            if tier is None:
                # Player's computed BB/100 from our sample is below 5;
                # they're in the good list but our sample is too small.
                # Use the lowest tier as fallback.
                tier = TIERS[0][2]

            # Classify spot
            spot = _classify_spot(obs, facing_bet)

            # Track agreement
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

            # Disagreement weighted by pot
            if not exact_agree:
                tier_disagree_pot_weighted[tier] += pot_dollars
            tier_total_pot[tier] += pot_dollars

            # Track action distributions per spot
            spot_model_actions[tier][spot][m_action] += 1
            spot_player_actions[tier][spot][actual] += 1

        if (r_idx + 1) % 5000 == 0:
            print(f'  Processed {r_idx + 1}/{len(all_results)} hands, '
                  f'{n_total} decision points so far...')

    analysis_time = time.time() - t1
    print(f'Analysis complete in {analysis_time:.1f}s')

    # ── Print results ────────────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print('BENCHMARK: MODEL vs WINNING PLAYERS')
    print('=' * 78)

    if split_models:
        print(f'Models: FB={os.path.basename(fb_path)}, CT={os.path.basename(ct_path)}')
    else:
        print(f'Model: {os.path.basename(fb_path)}')

    print(f'Hands parsed: {len(all_results)}, '
          f'Decision points: {n_total} '
          f'(facing bet: {n_facing_bet}, checked to: {n_checked_to})')
    print(f'Preflop skipped: {n_skipped_preflop}, '
          f'No cards skipped: {n_skipped_no_cards}')

    # ── Overall agreement ────────────────────────────────────────────────────
    print(f'\n--- Overall Agreement ---')
    if n_total > 0:
        print(f'  Exact action match:  {n_agree:>6}/{n_total:<6} = {100*n_agree/n_total:.1f}%')
        print(f'  Category match:      {n_agree_cat:>6}/{n_total:<6} = {100*n_agree_cat/n_total:.1f}%')
        print(f'    (Category: passive=fold, flat=call/check, aggressive=raise/allin)')
    else:
        print('  No decision points found.')
        return

    # ── Per-tier agreement ───────────────────────────────────────────────────
    print(f'\n--- Agreement by Win-Rate Tier ---')
    print(f'{"Tier":<16} {"DPs":>6} {"Exact":>7} {"Category":>9} {"Disagree$/pot$":>14}')
    print('-' * 58)
    for _, _, label in TIERS:
        t = tier_total.get(label, 0)
        if t < args.min_sample:
            print(f'{label:<16} {t:>6}   (insufficient sample)')
            continue
        a = tier_agree.get(label, 0)
        ac = tier_agree_category.get(label, 0)
        dpw = tier_disagree_pot_weighted.get(label, 0)
        tpot = tier_total_pot.get(label, 0)
        dpw_pct = (dpw / tpot * 100) if tpot > 0 else 0
        print(f'{label:<16} {t:>6} {100*a/t:>6.1f}% {100*ac/t:>8.1f}% {dpw_pct:>13.1f}%')

    # ── Per-street agreement ─────────────────────────────────────────────────
    print(f'\n--- Agreement by Street ---')
    print(f'{"Street":<10} ', end='')
    for _, _, label in TIERS:
        print(f'  {label:<16}', end='')
    print()
    print('-' * 62)
    for street in ['flop', 'turn', 'river']:
        print(f'{street:<10} ', end='')
        for _, _, label in TIERS:
            t = street_total.get(label, {}).get(street, 0)
            a = street_agree.get(label, {}).get(street, 0)
            if t >= args.min_sample:
                print(f'  {100*a/t:>5.1f}% ({t:>5})', end='')
            elif t > 0:
                print(f'  {100*a/t:>5.1f}% ({t:>5})', end='')
            else:
                print(f'       -       ', end='')
        print()

    # ── Per-spot breakdown ───────────────────────────────────────────────────
    print(f'\n--- Agreement by Spot Type ---')
    # Aggregate across tiers first
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

    print(f'{"Spot":<32} {"DPs":>6} {"Agree":>7}  {"Model top action":>16}  {"Player top action":>17}')
    print('-' * 88)

    for spot_key in ['facing_bet_air', 'facing_bet_draw', 'facing_bet_pair',
                     'facing_bet_strong', 'checked_to_air', 'checked_to_draw',
                     'checked_to_pair', 'checked_to_strong']:
        label = SPOT_NAMES.get(spot_key, spot_key)
        t = agg_spot_total.get(spot_key, 0)
        if t == 0:
            continue
        a = agg_spot_agree.get(spot_key, 0)

        # Most common model action
        m_acts = agg_spot_model.get(spot_key, {})
        m_top = max(m_acts, key=m_acts.get) if m_acts else -1
        m_pct = (m_acts[m_top] / t * 100) if m_top >= 0 else 0

        # Most common player action
        p_acts = agg_spot_player.get(spot_key, {})
        p_top = max(p_acts, key=p_acts.get) if p_acts else -1
        p_pct = (p_acts[p_top] / t * 100) if p_top >= 0 else 0

        m_name = ACTION_NAMES[m_top] if 0 <= m_top < len(ACTION_NAMES) else '?'
        p_name = ACTION_NAMES[p_top] if 0 <= p_top < len(ACTION_NAMES) else '?'

        print(f'{label:<32} {t:>6} {100*a/t:>6.1f}%  '
              f'{m_name:>8} ({m_pct:>4.1f}%)  '
              f'{p_name:>9} ({p_pct:>4.1f}%)')

    # ── Action distribution comparison ───────────────────────────────────────
    print(f'\n--- Action Distribution: Model vs Winners ---')
    print(f'{"":>32}  {"Fold":>7} {"Call":>7} {"Rse1/2":>7} {"RsePot":>7} {"AllIn":>7}')
    print('-' * 82)

    for spot_key in ['facing_bet_air', 'facing_bet_draw', 'facing_bet_pair',
                     'facing_bet_strong', 'checked_to_air', 'checked_to_draw',
                     'checked_to_pair', 'checked_to_strong']:
        label = SPOT_NAMES.get(spot_key, spot_key)
        t = agg_spot_total.get(spot_key, 0)
        if t == 0:
            continue

        m_acts = agg_spot_model.get(spot_key, {})
        p_acts = agg_spot_player.get(spot_key, {})

        # Model distribution
        m_dist = '  Model: '
        m_str = ''
        for a in range(5):
            pct = (m_acts.get(a, 0) / t * 100) if t > 0 else 0
            m_str += f' {pct:>6.1f}%'
        print(f'{label + " (model)":<32} {m_str}')

        # Player distribution
        p_str = ''
        for a in range(5):
            pct = (p_acts.get(a, 0) / t * 100) if t > 0 else 0
            p_str += f' {pct:>6.1f}%'
        print(f'{"  (winners)":<32} {p_str}')

    # ── Estimated model strength ─────────────────────────────────────────────
    print(f'\n--- Estimated Model Strength ---')
    print('  (Higher agreement with stronger tiers = better model)')

    # Compute a weighted score: agreement with 15+ BB/100 players matters most
    tier_weights = {TIERS[0][2]: 1.0, TIERS[1][2]: 1.5, TIERS[2][2]: 2.0}
    weighted_agree = 0.0
    weighted_total = 0.0
    for _, _, label in TIERS:
        t = tier_total.get(label, 0)
        a = tier_agree.get(label, 0)
        w = tier_weights.get(label, 1.0)
        if t >= args.min_sample:
            weighted_agree += (a / t) * w
            weighted_total += w

    if weighted_total > 0:
        weighted_score = weighted_agree / weighted_total * 100
        print(f'  Weighted agreement score: {weighted_score:.1f}%')
        print(f'    (Weights: 5-10=1.0x, 10-15=1.5x, 15+=2.0x)')

    # Aggressive alignment check: are we too passive or too aggressive?
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

    # Disagreement cost
    total_dpw = sum(tier_disagree_pot_weighted.values())
    total_tpot = sum(tier_total_pot.values())
    if total_tpot > 0:
        print(f'  Disagreement cost: {total_dpw:.2f} (pot-weighted) of '
              f'${total_tpot:.2f} total pot')
        print(f'    = {total_dpw/total_tpot*100:.1f}% of pot at risk in disagreements')

    print(f'\n{"=" * 78}')
    print('Done.')




if __name__ == '__main__':
    main()
