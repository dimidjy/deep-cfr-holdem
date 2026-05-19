"""Check tier selector with biased-toward-call play (more realistic than random)."""
import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from collections import Counter

from src.engine import EnhancedPokerEngine
from src.range_equity import select_threat_tier

random.seed(0)
np.random.seed(0)

eng = EnhancedPokerEngine(
    num_players=6,
    add_position_features=True, add_pot_odds_features=True,
    add_hand_strength_features=True, add_stack_features=True,
    add_equity_features=True, add_street_features=True,
    add_game_state_features=True, add_betting_history_features=True,
    behavioral_stats=True, betting_line=True, threat_range_equity=True,
)

EQUITY_START = 8 + 3 + 5 + 2
BH_HIST_START = EQUITY_START + 9 + 4 + 6
BEHAVIORAL_START = BH_HIST_START + 5

# Bias action: 70% call (action 1), 20% raise_half (2), 10% fold (0)
# This is closer to NL5 fish behavior than fully random
def biased_action(legal):
    weights = []
    for a in legal:
        if a == 0: weights.append(0.10)
        elif a == 1: weights.append(0.70)
        elif a == 2: weights.append(0.15)
        else:        weights.append(0.05)
    w = np.array(weights); w /= w.sum()
    return int(np.random.choice(legal, p=w))


tier_counter = Counter()
te_by_tier = {0: [], 1: [], 2: [], 3: []}
N_HANDS = 200
for _ in range(N_HANDS):
    eng.reset()
    steps = 0
    while not eng.is_done() and steps < 100:
        obs = eng.get_observation()
        legal = eng.get_legal_actions()
        if not legal: break
        n_pub = len(eng._state.public_cards)
        if n_pub >= 3:
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
            te_by_tier[tier].append(float(obs[53]))
        try:
            eng.step(biased_action(legal))
        except Exception:
            break
        steps += 1

total = sum(tier_counter.values())
print(f'Tier distribution under BIASED-CALL play ({total} postflop spots):')
for t in sorted(tier_counter):
    n = tier_counter[t]
    avg_te = np.mean(te_by_tier[t]) if te_by_tier[t] else 0
    print(f'  tier {t}: {n:5d}  ({100*n/total:5.1f}%)   avg threat_eq = {avg_te:.3f}')

# Re-test with NEVER-RAISE play (always call/check/fold) — should give mostly tier 0
print('\nNEVER-RAISE play (50% call, 50% fold):')
tier_counter = Counter()
N_HANDS = 200
for _ in range(N_HANDS):
    eng.reset()
    steps = 0
    while not eng.is_done() and steps < 100:
        obs = eng.get_observation()
        legal = eng.get_legal_actions()
        if not legal: break
        n_pub = len(eng._state.public_cards)
        if n_pub >= 3:
            tier = select_threat_tier(
                num_raises_norm=float(obs[BEHAVIORAL_START + 4]),
                max_bet_sizing=float(obs[BEHAVIORAL_START + 3]),
                facing_3bet_plus=float(obs[BH_HIST_START + 2]),
                pot_was_raised=float(obs[BH_HIST_START + 3]),
                board_flush_texture=float(obs[EQUITY_START + 6]),
                board_straight_texture=float(obs[EQUITY_START + 7]))
            tier_counter[tier] += 1
        # Always call (action 1), never raise
        action = 1 if 1 in legal else legal[0]
        try:
            eng.step(int(action))
        except Exception:
            break
        steps += 1
total = sum(tier_counter.values())
for t in sorted(tier_counter):
    n = tier_counter[t]
    print(f'  tier {t}: {n:5d}  ({100*n/total:5.1f}%)')
