"""Trace which code path produces threat_range_equity values.

Adds verbose probes into the threat-eq computation to see:
  1. Which tier is selected per spot
  2. Whether DB lookup fires (board-aware vs plain vs miss)
  3. The raw equity from each path before min() floor
  4. The final returned value
"""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from collections import Counter

from src.engine import EnhancedPokerEngine
from src.range_equity import (
    compute_threat_range_equity, select_threat_tier
)
from src.opp_range_lookup import get_default_lookup

random.seed(42)
np.random.seed(42)

eng = EnhancedPokerEngine(
    num_players=6,
    add_position_features=True, add_pot_odds_features=True,
    add_hand_strength_features=True, add_stack_features=True,
    add_equity_features=True, add_street_features=True,
    add_game_state_features=True, add_betting_history_features=True,
    behavioral_stats=True, betting_line=True, threat_range_equity=True,
)

# Indices (54-dim, 6p)
POS = 8
EQUITY_START = POS + 3 + 5 + 2  # = 18
BH_HIST_START = EQUITY_START + 9 + 4 + 6  # = 37
BEHAVIORAL_START = BH_HIST_START + 5  # = 42
THREAT_IDX = 53

db = get_default_lookup('data/opp_range.sqlite')

samples = []
tier_counter = Counter()
N_HANDS = 100
for h in range(N_HANDS):
    eng.reset()
    steps = 0
    while not eng.is_done() and steps < 100:
        obs = eng.get_observation()
        legal = eng.get_legal_actions()
        if not legal:
            break
        n_pub = len(eng._state.public_cards)
        if n_pub >= 3:
            hs = float(obs[18])
            te = float(obs[THREAT_IDX])

            num_raises_norm = float(obs[BEHAVIORAL_START + 4])
            max_bet_sizing  = float(obs[BEHAVIORAL_START + 3])
            facing_3bet     = float(obs[BH_HIST_START + 2])
            pot_was_raised  = float(obs[BH_HIST_START + 3])
            bft = float(obs[EQUITY_START + 6])
            bst = float(obs[EQUITY_START + 7])
            tier = select_threat_tier(
                num_raises_norm=num_raises_norm,
                max_bet_sizing=max_bet_sizing,
                facing_3bet_plus=facing_3bet,
                pot_was_raised=pot_was_raised,
                board_flush_texture=bft,
                board_straight_texture=bst)
            tier_counter[tier] += 1

            # Recompute tier-only equity to see if THE TIER itself gives low values
            try:
                ps = eng._cur_ps()
                from src.engine.fast_enhanced_engine import (
                    _pkrs_card_rank, _pkrs_card_suit)
                hole_strs = [_pkrs_card_rank(c) + _pkrs_card_suit(c).lower()
                             for c in ps.hand[:2]]
                board_strs = [_pkrs_card_rank(c) + _pkrs_card_suit(c).lower()
                              for c in eng._state.public_cards]
            except Exception as e:
                hole_strs, board_strs = None, None

            tier_eq = None
            if hole_strs and board_strs:
                tier_eq = compute_threat_range_equity(
                    hole_strs, board_strs, tier, n_samples=300)

            samples.append({
                'street': n_pub,
                'hs': hs,
                'te': te,
                'tier': tier,
                'tier_eq': tier_eq,
                'num_raises': num_raises_norm * 6,
                'pot_was_raised': pot_was_raised,
            })

        try:
            eng.step(int(random.choice(legal)))
        except Exception:
            break
        steps += 1

print(f'Collected {len(samples)} postflop spots')
print(f'\nTier distribution:')
for t in sorted(tier_counter):
    pct = 100 * tier_counter[t] / sum(tier_counter.values())
    print(f'  tier {t}: {tier_counter[t]:4d}  ({pct:.1f}%)')

# By-tier statistics
print(f'\nBy-tier obs[53] threat_eq stats:')
print(f'  {"tier":<6} {"n":<6} {"hs_mean":<10} {"te_mean":<10} {"diff_mean":<10}')
by_tier = {}
for s in samples:
    by_tier.setdefault(s['tier'], []).append(s)
for tier in sorted(by_tier):
    rows = by_tier[tier]
    hs = np.array([r['hs'] for r in rows])
    te = np.array([r['te'] for r in rows])
    diff = te - hs
    print(f'  {tier:<6} {len(rows):<6} {hs.mean():<10.3f} {te.mean():<10.3f} '
          f'{diff.mean():<+10.3f}')

# How often does tier_eq == te (DB miss case)?
print(f'\nDB hit detection:')
no_tier_data = sum(1 for s in samples if s['tier_eq'] is None)
print(f'  spots where tier_eq could not be computed: {no_tier_data}')

# Examples of spots where te is way lower than hs
print(f'\nWorst threat_eq cases (te much lower than hs):')
samples.sort(key=lambda s: s['te'] - s['hs'])
for s in samples[:10]:
    print(f"  street={s['street']}  tier={s['tier']}  "
          f"hs={s['hs']:.3f}  te={s['te']:.3f}  "
          f"tier_eq_recomputed={s['tier_eq']:.3f}  "
          f"raises={s['num_raises']:.1f}  pot_was_raised={s['pot_was_raised']:.1f}")

print(f'\nBest threat_eq cases (te > hs):')
for s in sorted(samples, key=lambda s: -(s['te'] - s['hs']))[:5]:
    print(f"  street={s['street']}  tier={s['tier']}  "
          f"hs={s['hs']:.3f}  te={s['te']:.3f}  "
          f"tier_eq_recomputed={s['tier_eq']:.3f}  "
          f"raises={s['num_raises']:.1f}  pot_was_raised={s['pot_was_raised']:.1f}")
