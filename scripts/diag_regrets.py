"""
Diagnostic: run external sampling traversals and show regrets per hand type.
No trained model needed — uses random advantage network (iteration 1 equivalent).

Run: python3 scripts/diag_regrets.py
"""
import sys, os, warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from collections import defaultdict
from src.engine import EnhancedPokerEngine
from src.training import DeepCFR

REWARD_SCALE = 100.0
REGRET_CLIP  = 2.0
NUM_PLAYERS  = 6
NUM_TRAV     = 300   # traversals per player_idx=0

RANK_MAP = {'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
            'T':10,'J':11,'Q':12,'K':13,'A':14}

def classify(cards):
    if len(cards) < 2: return None
    r1, r2 = RANK_MAP.get(cards[0][0],0), RANK_MAP.get(cards[1][0],0)
    s1, s2 = cards[0][1], cards[1][1]
    hi, lo = max(r1,r2), min(r1,r2)
    is_pair   = r1 == r2
    is_suited = s1 == s2
    if is_pair:
        if hi==14: return 'AA'
        if hi==13: return 'KK'
        if hi==12: return 'QQ'
        return 'pair'
    if hi==14 and lo>=13: return 'AK'
    if hi<=8 and (hi-lo)>=3 and not is_suited:
        if hi==7 and lo==2: return '72o'
        return 'trash'
    return 'other'

def make_engine():
    return EnhancedPokerEngine(
        num_players=NUM_PLAYERS,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
    )

def regret_match(adv, legal, epsilon=0.05):
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    if s > 0:
        strategy[legal] = pos / s
    else:
        strategy[legal] = 1.0 / len(legal)
    strategy[legal] = (1-epsilon)*strategy[legal] + epsilon/len(legal)
    return strategy

def pick_random(legal):
    return int(np.random.choice(legal))

def run_to_terminal(eng):
    while not eng.is_done():
        legal = eng.get_legal_actions()
        if not legal: break
        eng.step(pick_random(legal))
    rw = eng.get_rewards()
    return float(rw[eng.agents[0]])

def main():
    engine = make_engine()
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()
    engine.close()

    # Random-init advantage network (simulates iteration 1)
    deep_cfr = DeepCFR(
        observation_size=obs_size,
        action_size=act_size,
        num_players=NUM_PLAYERS,
        hidden_layers=[512, 512, 256],
        device='cpu',
    )
    adv_net = deep_cfr.advantage_networks[0].eval()

    # Collect regrets per hand category
    regrets_by_hand = defaultdict(lambda: defaultdict(list))  # hand → action → [regret]
    rewards_by_hand = defaultdict(lambda: defaultdict(list))  # hand → action → [branch_reward]
    hand_counts = defaultdict(int)

    ACTION_NAMES = {0:'fold', 1:'call', 2:'raise_h', 3:'raise_f', 4:'raise_2x', 5:'allin'}

    for trav in range(NUM_TRAV):
        eng1 = make_engine()
        pre_game_rng  = eng1._get_game().np_random.get_state()
        pre_dealer_id = eng1._get_game().dealer_id
        eng1.reset()

        pre_actions = []
        first_obs   = None
        first_legal = None
        hole_cards  = None

        # Phase 1: play to player 0's first decision
        while not eng1.is_done():
            ca  = eng1.current_agent
            ai  = eng1.agents.index(ca)
            obs = eng1.get_observation()
            legal = eng1.get_legal_actions()
            if not legal: break
            if ai == 0:
                first_obs   = obs.copy()
                first_legal = [int(a) for a in legal]
                hole_cards  = eng1.get_player_cards(ca)
                break
            action = pick_random(legal)
            pre_actions.append(action)
            eng1.step(action)
        eng1.close()

        if first_obs is None:
            continue

        cat = classify(hole_cards) if hole_cards else None
        hand_counts[cat] += 1

        # Phase 2: branch over all legal actions
        branch_rewards = {}
        for ba in first_legal:
            eng2 = make_engine()
            eng2._get_game().np_random.set_state(pre_game_rng)
            eng2._get_game().dealer_id = pre_dealer_id
            eng2.reset()
            for stored in pre_actions:
                eng2.step(stored)
            eng2.step(ba)
            reward = run_to_terminal(eng2)
            branch_rewards[ba] = reward

        # Phase 3: compute exact regrets
        with torch.no_grad():
            obs_t  = torch.from_numpy(first_obs.astype(np.float32)).unsqueeze(0)
            adv_np = adv_net(obs_t).numpy()[0]
        strategy = regret_match(adv_np, first_legal)
        ev = sum(strategy[a] * branch_rewards[a] / REWARD_SCALE for a in first_legal)

        for a in first_legal:
            r = float(np.clip(branch_rewards[a]/REWARD_SCALE - ev, -REGRET_CLIP, REGRET_CLIP))
            regrets_by_hand[cat][a].append(r)
            rewards_by_hand[cat][a].append(branch_rewards[a])

    # Report
    print(f"\n{'='*65}")
    print(f"  EXACT REGRET DIAGNOSTICS  ({NUM_TRAV} traversals, player_idx=0)")
    print(f"  Regret > 0 → should do this action more")
    print(f"  Regret < 0 → should do this action less")
    print(f"{'='*65}\n")

    for cat in ['AA','KK','QQ','AK','72o','trash','pair','other']:
        if cat not in regrets_by_hand:
            continue
        n = hand_counts[cat]
        print(f"  {cat}  (n={n})")
        all_actions = sorted(regrets_by_hand[cat].keys())
        for a in all_actions:
            regs  = regrets_by_hand[cat][a]
            rews  = rewards_by_hand[cat][a]
            mean_r  = np.mean(regs)
            std_r   = np.std(regs)
            mean_rew = np.mean(rews)
            aname = ACTION_NAMES.get(a, str(a))
            direction = "↑ play more" if mean_r > 0.05 else ("↓ fold/avoid" if mean_r < -0.05 else "≈ neutral")
            print(f"    action={aname:8s}: regret={mean_r:+.3f} ±{std_r:.3f}  "
                  f"branch_reward={mean_rew:+6.1f}  {direction}")
        print()

    print(f"{'='*65}")
    print("\nKEY CHECK: For AA, fold regret should be NEGATIVE (avoid fold)")
    print("           For 72o/trash, fold regret should be POSITIVE (fold more)\n")

if __name__ == '__main__':
    main()
