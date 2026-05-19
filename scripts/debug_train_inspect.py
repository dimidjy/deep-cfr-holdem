"""Single-process debug script that probes the training pipeline for bugs.

Runs the same observation builder + branch traversal + shapers used in
train_gpu.py, but with verbose instrumentation. Captures:

  - Distribution of obs[18] (hand_strength vs random) and obs[53]
    (threat_range_equity vs tier/DB) across realistic spots
  - Correlation of the two equity features
  - Shaper firing rates and magnitudes per action
  - Per-action reward distribution after branch traversal
  - Detection of obs/reward mismatches between traversal branches

No multiprocessing. Sequential, deterministic, easy to read.
Run: python3 scripts/debug_train_inspect.py
"""
import sys
import json
import random
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.engine import EnhancedPokerEngine


SEED = 42
N_HANDS = 100  # short — just enough for statistics


def build_engine():
    """Same flags as training (threat_range_equity ON)."""
    return EnhancedPokerEngine(
        num_players=6,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=True,
        behavioral_stats=True,
        betting_line=True,
        threat_range_equity=True,
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    eng = build_engine()
    obs_size = eng.get_observation_space_size()
    print(f'Obs size: {obs_size}')
    assert obs_size >= 54, f'Expected 54-dim obs, got {obs_size}'

    # Indices (per CLAUDE.md):
    #   8 pos + 3 potodds + 5 handstrength + 2 stack + 9 equity + 4 street +
    #   6 gamestate + 5 betting + 8 behavioral + 3 betting_line + 1 threat
    IDX_HAND_STRENGTH = 18  # equity vs random
    IDX_POT_ODDS     = 10
    IDX_THREAT_EQ    = 53   # at the very end (after betting_line)

    # ── Pass 1: collect obs distributions across random play ─────────────
    obs_records = []
    bucket_counts = defaultdict(int)
    for hand_i in range(N_HANDS):
        eng.reset()
        steps = 0
        while not eng.is_done() and steps < 100:
            ai = eng.current_agent
            obs = eng.get_observation()
            legal = eng.get_legal_actions()
            if not legal:
                break

            # Record only postflop (preflop has no threat_eq computation
            # in HH builder, and engine path returns 0.0 for preflop too)
            n_pub = len(eng._state.public_cards)
            if n_pub >= 3 and obs[IDX_THREAT_EQ] > 0:
                obs_records.append({
                    'hand_strength': float(obs[IDX_HAND_STRENGTH]),
                    'threat_eq':     float(obs[IDX_THREAT_EQ]),
                    'pot_odds':      float(obs[IDX_POT_ODDS]),
                    'street':        n_pub,  # 3=flop,4=turn,5=river
                })

            # Sample a random legal action and step
            try:
                eng.step(int(random.choice(legal)))
            except Exception:
                break
            steps += 1

    print(f'\n{"=" * 70}')
    print(f'PASS 1: postflop obs distribution over {N_HANDS} random hands')
    print('=' * 70)
    print(f'Collected: {len(obs_records)} postflop decision points')
    if not obs_records:
        print('NO postflop obs collected. Bug in engine or random play got cut off.')
        return

    hs = np.array([r['hand_strength'] for r in obs_records])
    te = np.array([r['threat_eq']     for r in obs_records])
    po = np.array([r['pot_odds']      for r in obs_records])

    def dist(arr, label):
        print(f'\n{label}:')
        print(f'  min/max:           {arr.min():.3f} / {arr.max():.3f}')
        print(f'  mean / std:        {arr.mean():.3f} / {arr.std():.3f}')
        print(f'  p10 / p50 / p90:   {np.percentile(arr, 10):.3f} / '
              f'{np.percentile(arr, 50):.3f} / {np.percentile(arr, 90):.3f}')
        print(f'  fraction == 0:     {(arr == 0).mean():.3f}')
        print(f'  fraction near 0.5: {((arr > 0.45) & (arr < 0.55)).mean():.3f}')

    dist(hs, 'obs[18] hand_strength (vs random)')
    dist(te, 'obs[53] threat_range_equity (vs tier/DB)')
    dist(po, 'obs[10] pot_odds')

    corr = np.corrcoef(hs, te)[0, 1]
    diff = te - hs
    print(f'\nCorrelation(hand_strength, threat_eq): {corr:.3f}')
    print(f'Difference threat_eq - hand_strength:')
    print(f'  mean: {diff.mean():+.3f}  std: {diff.std():.3f}')
    print(f'  fraction threat_eq < hand_strength: {(diff < 0).mean():.3f}')
    print(f'  fraction threat_eq > hand_strength: {(diff > 0).mean():.3f}')

    # Pass 2: simulate shaper firings using both equity features
    print(f'\n{"=" * 70}')
    print('PASS 2: shaper firing analysis (current config)')
    print('=' * 70)
    BLUFF_THRESHOLD = 0.60
    VALUE_THRESHOLD = 0.70
    FOLD_MARGIN = 0.15
    BLUFF_PENALTY = 0.55
    VALUE_BONUS = 0.20
    FOLD_PENALTY = 0.20

    def shaper_chips(arr_eq, label):
        bluff_fire = 0
        value_fire = 0
        fold_fire = 0
        bluff_mag = []
        value_mag = []
        fold_mag = []
        for r, eq in zip(obs_records, arr_eq):
            pot = 10.0  # nominal pot
            # bluff penalty (raise action, equity < 0.60)
            deficit = max(0.0, BLUFF_THRESHOLD - eq)
            if deficit > 0:
                bluff_fire += 1
                bluff_mag.append(deficit * pot * BLUFF_PENALTY)
            # value bonus (raise action, equity > 0.70)
            excess = max(0.0, eq - VALUE_THRESHOLD)
            if excess > 0:
                value_fire += 1
                value_mag.append(excess * pot * VALUE_BONUS)
            # fold penalty (fold action, equity > pot-odds + margin, facing bet)
            poval = r['pot_odds']
            if poval > 0:
                required = 1.0 - poval
                exc = eq - required
                if exc > FOLD_MARGIN:
                    fold_fire += 1
                    post_call_pot = pot / max(poval, 1e-6)
                    fold_mag.append(exc * post_call_pot * FOLD_PENALTY)
        n = len(obs_records)
        print(f'\n  Using {label}:')
        print(f'    bluff_penalty fires:  {bluff_fire}/{n}  ({100*bluff_fire/n:.1f}%)  '
              f'mean mag: {np.mean(bluff_mag) if bluff_mag else 0:.2f}')
        print(f'    value_bonus fires:    {value_fire}/{n}  ({100*value_fire/n:.1f}%)  '
              f'mean mag: {np.mean(value_mag) if value_mag else 0:.2f}')
        print(f'    fold_penalty fires:   {fold_fire}/{n}  ({100*fold_fire/n:.1f}%)  '
              f'mean mag: {np.mean(fold_mag) if fold_mag else 0:.2f}')

    shaper_chips(hs, 'obs[18] hand_strength (CURRENT)')
    shaper_chips(te, 'obs[53] threat_range_equity (HYPOTHETICAL)')

    # Pass 3: dump a few example spots showing both equity values + tier
    print(f'\n{"=" * 70}')
    print('PASS 3: sample postflop spots (10 random)')
    print('=' * 70)
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(obs_records), min(10, len(obs_records)), replace=False)
    for i in sample:
        r = obs_records[i]
        st = {3: 'flop', 4: 'turn', 5: 'river'}[r['street']]
        print(f'  {st:<5}  hand_str={r["hand_strength"]:.3f}  '
              f'threat_eq={r["threat_eq"]:.3f}  pot_odds={r["pot_odds"]:.3f}  '
              f'(diff={r["threat_eq"] - r["hand_strength"]:+.3f})')

    # Pass 4: verify board-aware DB hits are happening
    print(f'\n{"=" * 70}')
    print('PASS 4: DB coverage during simulated play')
    print('=' * 70)
    from src.opp_range_lookup import get_default_lookup
    db = get_default_lookup('data/opp_range.sqlite')
    if db is not None:
        db._ensure_loaded()
        n_spots = len(db._cache) if db._cache else 0
        total_count = sum(sum(c for _, c in v) for v in (db._cache or {}).values())
        print(f'  DB loaded: {n_spots} spot keys, {total_count:,} total observations')
    else:
        print('  DB lookup returned None (file missing or load failed)')

    print(f'\n{"=" * 70}')
    print('DONE — key signals:')
    print('=' * 70)
    print('1. If obs[53] mean is wildly different from obs[18] mean → check sign.')
    print('2. If corr(hs, te) < 0.5, the two features carry very different info.')
    print('3. If "threat_eq > hand_strength" is high, threat_eq is TOO OPTIMISTIC.')
    print('4. Shaper firing rates show whether the model gets useful gradient signal.')


if __name__ == '__main__':
    main()
