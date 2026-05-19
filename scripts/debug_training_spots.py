"""Deep investigation of training spots with the threat_range_equity feature.

Captures REAL spots from simulated hands and probes:
  1. Full obs vector (with feature breakdown by block)
  2. Threat-tier selection + path-by-path equity values (tier MC, DB MC, final)
  3. Shaper firings: which fire, magnitudes, sign
  4. Hero model strategy on the spot
  5. Pool model strategy on the same spot (verifies 53→54 padding is correct)
  6. Sensitivity test: does hero model strategy change when obs[53] varies?

Usage:
  python3 scripts/debug_training_spots.py \\
    --hero models/v52/checkpoint_iter_4100.pt \\
    --pool models/best/checkpoint_iter_4000.pt \\
    --n-spots 8
"""
import argparse
import sys
import random
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.engine import EnhancedPokerEngine
from src.range_equity import (
    select_threat_tier, compute_threat_range_equity, get_threat_range_equity
)
from src.opp_range_lookup import get_default_lookup

# Obs layout indices (6p, 54-dim)
POS_END = 8
IDX_POT_ODDS = 10
IDX_HAND_STRENGTH = 18
EQUITY_START = 18
GAME_STATE_START = 18 + 9 + 4   # 31
BH_HIST_START = GAME_STATE_START + 6  # 37
BEHAVIORAL_START = BH_HIST_START + 5  # 42
BETTING_LINE_START = BEHAVIORAL_START + 8  # 50
THREAT_IDX = 53

# Shaper thresholds (current train_gpu.py)
BLUFF_THRESHOLD = 0.60
VALUE_THRESHOLD = 0.70
FOLD_MARGIN = 0.15


def _load_strategy_net(path: Path, target_obs_size: int, act_size: int,
                       hidden=[512, 512, 256], dropout=0.15) -> nn.Sequential:
    """Load a strategy network, padding 53-dim source to target_obs_size."""
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    state = ckpt['strategy_networks'][0]
    src_obs = int(state['network.0.weight'].shape[1])

    # Pad weights if source is smaller
    if src_obs < target_obs_size:
        w = state['network.0.weight']
        pad = torch.zeros(w.shape[0], target_obs_size - src_obs, dtype=w.dtype)
        state = dict(state)
        state['network.0.weight'] = torch.cat([w, pad], dim=1)
        print(f'  Loaded {path.name}: padded {src_obs} → {target_obs_size} '
              f'(zero col for new dim)')
    elif src_obs > target_obs_size:
        state = dict(state)
        state['network.0.weight'] = state['network.0.weight'][:, :target_obs_size]
    else:
        print(f'  Loaded {path.name}: native {src_obs} dim')

    # Build network — mirror StrategyNetwork architecture from src/training/deep_cfr.py
    layers = []
    prev = target_obs_size
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.01))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, act_size))

    net = nn.Sequential(*layers)
    # State keys are like network.0.weight, network.1.bias, etc.
    # Our Sequential layers might not match exactly — try strict=False
    # Build mapping
    net_state = net.state_dict()
    out_state = {}
    for k, v in state.items():
        # State key like 'network.0.weight' → our key like '0.weight'
        new_k = k.replace('network.', '')
        if new_k in net_state and net_state[new_k].shape == v.shape:
            out_state[new_k] = v
    missing = set(net_state.keys()) - set(out_state.keys())
    if missing:
        print(f'  WARNING: {len(missing)} keys missing from state load: '
              f'{list(missing)[:3]}...')
    net.load_state_dict(out_state, strict=False)
    net.eval()
    return net, src_obs


def _strategy_from_logits(logits, legal):
    """Softmax over legal actions only (mimic regret_match's masked behavior)."""
    masked = np.full_like(logits, -1e9)
    for a in legal:
        masked[a] = logits[a]
    exp = np.exp(masked - masked.max())
    exp = exp * np.array([1.0 if a in legal else 0.0 for a in range(len(logits))])
    s = exp.sum()
    if s > 0:
        return exp / s
    out = np.zeros_like(logits)
    for a in legal:
        out[a] = 1.0 / len(legal)
    return out


def fmt_obs_block(name, values):
    """Format an obs block one-line."""
    vals_str = ', '.join(f'{v:.2f}' for v in values)
    return f'  {name:<18} [{vals_str}]'


def capture_spots(n_spots, seed=42):
    """Simulate hands with biased-call play, capture postflop spots."""
    random.seed(seed)
    np.random.seed(seed)
    eng = EnhancedPokerEngine(
        num_players=6,
        add_position_features=True, add_pot_odds_features=True,
        add_hand_strength_features=True, add_stack_features=True,
        add_equity_features=True, add_street_features=True,
        add_game_state_features=True, add_betting_history_features=True,
        behavioral_stats=True, betting_line=True, threat_range_equity=True,
    )

    def biased(legal):
        w = []
        for a in legal:
            if a == 0: w.append(0.05)
            elif a == 1: w.append(0.65)
            elif a == 2: w.append(0.20)
            else: w.append(0.05)
        w = np.array(w); w /= w.sum()
        return int(np.random.choice(legal, p=w))

    spots = []
    while len(spots) < n_spots:
        eng.reset()
        steps = 0
        while not eng.is_done() and steps < 100:
            obs = eng.get_observation()
            legal = eng.get_legal_actions()
            if not legal: break
            n_pub = len(eng._state.public_cards)
            if n_pub >= 3:
                # Capture spot details
                ps = eng._cur_ps()
                from src.engine.fast_enhanced_engine import (
                    _pkrs_card_rank, _pkrs_card_suit)
                try:
                    hole_strs = [_pkrs_card_rank(c) + _pkrs_card_suit(c).lower()
                                 for c in ps.hand[:2]]
                    board_strs = [_pkrs_card_rank(c) + _pkrs_card_suit(c).lower()
                                  for c in eng._state.public_cards]
                    pot = eng.get_pot_size()
                    spots.append({
                        'obs': obs.copy(),
                        'legal': list(legal),
                        'street': n_pub,
                        'hole': hole_strs,
                        'board': board_strs,
                        'pot_chips': pot,
                    })
                except Exception:
                    pass
                if len(spots) >= n_spots:
                    break
            try:
                eng.step(biased(legal))
            except Exception:
                break
            steps += 1
    return spots


def analyze_spot(spot, idx, hero_net, pool_net, pool_src_obs):
    """Deep-dive analysis of a single spot."""
    obs = spot['obs']
    legal = spot['legal']
    hole = spot['hole']
    board = spot['board']
    street_name = {3: 'flop', 4: 'turn', 5: 'river'}[spot['street']]
    pot = spot['pot_chips']

    print(f'\n{"=" * 78}')
    print(f'SPOT {idx + 1} — {street_name.upper()}  hole={"".join(hole)}  '
          f'board={"".join(board)}  pot={pot} chips')
    print('=' * 78)

    # ── Obs block breakdown ─────────────────────────────────────────
    hs = float(obs[IDX_HAND_STRENGTH])
    pot_odds = float(obs[IDX_POT_ODDS])
    threat_eq = float(obs[THREAT_IDX])
    bft = float(obs[EQUITY_START + 6])
    bst = float(obs[EQUITY_START + 7])
    num_raises = float(obs[BEHAVIORAL_START + 4])
    max_bet = float(obs[BEHAVIORAL_START + 3])
    facing_3bet = float(obs[BH_HIST_START + 2])
    pot_was_raised = float(obs[BH_HIST_START + 3])

    print(f'\nKey obs values:')
    print(f'  obs[18] hand_strength  = {hs:.3f}  (equity vs random)')
    print(f'  obs[53] threat_eq      = {threat_eq:.3f}  (equity vs tier/DB)')
    print(f'  obs[10] pot_odds       = {pot_odds:.3f}')
    print(f'  Difference (te − hs)   = {threat_eq - hs:+.3f}')

    # ── Tier selection trace ──────────────────────────────────────────
    tier = select_threat_tier(
        num_raises_norm=num_raises, max_bet_sizing=max_bet,
        facing_3bet_plus=facing_3bet, pot_was_raised=pot_was_raised,
        board_flush_texture=bft, board_straight_texture=bst)

    print(f'\nTier selection:')
    print(f'  num_raises_norm  = {num_raises:.3f}  (IGNORED — hand-wide leak)')
    print(f'  max_bet_sizing   = {max_bet:.3f}    (IGNORED — hand-wide leak)')
    print(f'  facing_3bet      = {facing_3bet:.3f}')
    print(f'  pot_was_raised   = {pot_was_raised:.3f}')
    print(f'  board_flush_tex  = {bft:.3f}')
    print(f'  board_straight_tex={bst:.3f}')
    print(f'  → tier           = {tier}  ({["random","top40%","top15%","top5%"][tier]})')

    # ── Equity path trace ─────────────────────────────────────────────
    tier_only_eq = compute_threat_range_equity(hole, board, tier, n_samples=300)
    db = get_default_lookup('data/opp_range.sqlite')
    final_eq = get_threat_range_equity(
        hole, board, tier, spot_key=None, db_lookup=db, n_samples=300)

    print(f'\nEquity path:')
    print(f'  tier-only MC (300 samples) = {tier_only_eq:.3f}')
    print(f'  final (DB→tier floor)      = {final_eq:.3f}')
    print(f'  obs[53] in vector          = {threat_eq:.3f}  '
          f'(may differ from re-computed due to MC variance)')

    # ── Shaper firing analysis ──────────────────────────────────────
    print(f'\nShaper firings (per-action chips added/subtracted):')
    print(f'  Using obs[18] hand_strength = {hs:.3f} (CURRENT shaper signal)')
    fired = []
    for action_name, action_id in [('Fold', 0), ('Call', 1), ('Raise½', 2), ('RaisePot', 3)]:
        if action_id not in legal:
            continue
        bp = vc = cb = fp = 0.0
        # bluff_penalty (raise actions, equity < threshold)
        if action_id >= 2 and hs < BLUFF_THRESHOLD:
            bp = (BLUFF_THRESHOLD - hs) * pot * 0.5
        # value_bonus (raise actions, equity > threshold)
        if action_id >= 2 and hs > VALUE_THRESHOLD:
            vc = (hs - VALUE_THRESHOLD) * pot * 0.2
        # call_bonus (call, equity > required)
        if action_id == 1 and pot_odds > 0:
            req = 1.0 - pot_odds
            if hs > req:
                pcp = pot / max(pot_odds, 1e-6)
                cb = (hs - req) * pcp * 0.05
        # fold_penalty (fold, equity > required + margin)
        if action_id == 0 and pot_odds > 0:
            req = 1.0 - pot_odds
            if hs - req > FOLD_MARGIN:
                pcp = pot / max(pot_odds, 1e-6)
                fp = (hs - req) * pcp * 0.15
        net = -bp + vc + cb - fp
        sign = '+' if net >= 0 else '−'
        print(f'  {action_name:<10}  '
              f'bluff_pen={-bp:>+6.2f}  value_bon={+vc:>+6.2f}  '
              f'call_bon={+cb:>+6.2f}  fold_pen={-fp:>+6.2f}  '
              f'NET={sign}{abs(net):.2f}')
        fired.append(net)

    # If we recompute shaper signals using obs[53] instead, how do they change?
    print(f'\n  Using obs[53] threat_eq = {threat_eq:.3f} (HYPOTHETICAL)')
    for action_name, action_id in [('Fold', 0), ('Call', 1), ('Raise½', 2), ('RaisePot', 3)]:
        if action_id not in legal:
            continue
        bp = vc = cb = fp = 0.0
        eq = threat_eq
        if action_id >= 2 and eq < BLUFF_THRESHOLD:
            bp = (BLUFF_THRESHOLD - eq) * pot * 0.5
        if action_id >= 2 and eq > VALUE_THRESHOLD:
            vc = (eq - VALUE_THRESHOLD) * pot * 0.2
        if action_id == 1 and pot_odds > 0:
            req = 1.0 - pot_odds
            if eq > req:
                pcp = pot / max(pot_odds, 1e-6)
                cb = (eq - req) * pcp * 0.05
        if action_id == 0 and pot_odds > 0:
            req = 1.0 - pot_odds
            if eq - req > FOLD_MARGIN:
                pcp = pot / max(pot_odds, 1e-6)
                fp = (eq - req) * pcp * 0.15
        net = -bp + vc + cb - fp
        sign = '+' if net >= 0 else '−'
        print(f'  {action_name:<10}  '
              f'bluff_pen={-bp:>+6.2f}  value_bon={+vc:>+6.2f}  '
              f'call_bon={+cb:>+6.2f}  fold_pen={-fp:>+6.2f}  '
              f'NET={sign}{abs(net):.2f}')

    # ── Strategy comparison: hero vs pool ───────────────────────────
    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        hero_logits = hero_net(obs_t).numpy()[0]
        pool_logits = pool_net(obs_t).numpy()[0]
    hero_strat = _strategy_from_logits(hero_logits, legal)
    pool_strat = _strategy_from_logits(pool_logits, legal)

    print(f'\nStrategy comparison:')
    print(f'  {"Action":<10} {"Hero (v52)":>12} {"Pool (4000)":>12} {"Diff":>10}')
    action_names = ['Fold', 'Call', 'Raise½', 'RaisePot', 'AllIn']
    for a in legal:
        if a < len(action_names):
            name = action_names[a]
            diff = hero_strat[a] - pool_strat[a]
            print(f'  {name:<10} {hero_strat[a]:>11.3f}  {pool_strat[a]:>11.3f}  '
                  f'{diff:>+9.3f}')

    # Verify pool ignores obs[53] (padding test)
    obs_zero = obs.copy()
    obs_zero[THREAT_IDX] = 0.0  # what padding effectively gives
    obs_high = obs.copy()
    obs_high[THREAT_IDX] = 1.0
    with torch.no_grad():
        pool_zero = pool_net(
            torch.from_numpy(obs_zero.astype(np.float32)).unsqueeze(0)).numpy()[0]
        pool_high = pool_net(
            torch.from_numpy(obs_high.astype(np.float32)).unsqueeze(0)).numpy()[0]
        hero_zero = hero_net(
            torch.from_numpy(obs_zero.astype(np.float32)).unsqueeze(0)).numpy()[0]
        hero_high = hero_net(
            torch.from_numpy(obs_high.astype(np.float32)).unsqueeze(0)).numpy()[0]
    pool_drift = np.abs(pool_zero - pool_high).max()
    hero_drift = np.abs(hero_zero - hero_high).max()
    print(f'\nSensitivity to obs[53] (perturbed 0.0 → 1.0):')
    print(f'  Pool model logit change: {pool_drift:.6f}  '
          f'{"⚠️ pool reacts to obs[53]" if pool_drift > 1e-5 else "✓ pool ignores obs[53] (padding correct)"}')
    print(f'  Hero model logit change: {hero_drift:.6f}  '
          f'{"✓ hero uses obs[53]" if hero_drift > 0.001 else "⚠️ hero ignores obs[53]"}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hero', default='models/v52/checkpoint_iter_4100.pt')
    parser.add_argument('--pool', default='models/best/checkpoint_iter_4000.pt')
    parser.add_argument('--n-spots', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f'Loading models...')
    hero_net, hero_src = _load_strategy_net(Path(args.hero), 54, 6)
    pool_net, pool_src = _load_strategy_net(Path(args.pool), 54, 6)
    print(f'\n  Hero src dim: {hero_src} (54 = native, 53 = padded)')
    print(f'  Pool src dim: {pool_src} (54 = native, 53 = padded)')

    print(f'\nCapturing {args.n_spots} postflop spots (biased-call play)...')
    spots = capture_spots(args.n_spots, seed=args.seed)
    print(f'Captured {len(spots)} spots')

    for i, spot in enumerate(spots):
        analyze_spot(spot, i, hero_net, pool_net, pool_src)

    print(f'\n{"=" * 78}')
    print('SUMMARY checks:')
    print('=' * 78)
    print('  - Tier selection should match per-street signals (not hand-wide accumulators)')
    print('  - obs[53] should vary across spots (not stuck at one value)')
    print('  - obs[53] should be LOWER than obs[18] in aggression contexts (tier > 0)')
    print('  - obs[53] should be ≈ obs[18] in checked-down contexts (tier 0)')
    print('  - Pool model should IGNORE obs[53] (sensitivity ~0)')
    print('  - Hero model should USE obs[53] (sensitivity > 0.001)')


if __name__ == '__main__':
    main()
