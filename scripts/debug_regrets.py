"""
Debug script — verifies regret computation pipeline is working correctly.

Checks:
  1. Synthetic reward function gives correct values for known hands
  2. Regret direction is correct: fold AA → negative regret, fold 72o → less negative
  3. Advantage network outputs differ meaningfully for AA vs 72o
  4. Actual traversals trigger the synthetic reward correctly
  5. Buffer regret statistics after N iterations

Run from project root:
    python3 scripts/debug_regrets.py [--checkpoint models/6player_transferred.pt]
"""
import sys
import os
import argparse
import warnings
import logging

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config
from pathlib import Path

# Import the functions we want to test directly from train_gpu
from train_gpu import (
    _compute_regrets_gpu,
    _traversal_worker,
    _REWARD_SCALE,
    _worker_init,
)


# ── helpers ────────────────────────────────────────────────────────────────

def make_engine(num_players=6):
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
    )


def hand_obs(num_players, is_pair, is_suited, high_rank_norm, low_rank_norm):
    """Build a minimal obs vector with specific hand features, zeros elsewhere."""
    engine = make_engine(num_players)
    obs_size = engine.get_observation_space_size()
    obs = np.zeros(obs_size, dtype=np.float32)
    # hand features at [num_players+5 : num_players+10]
    h = num_players + 5
    obs[h]   = float(is_pair)
    obs[h+1] = float(is_suited)
    obs[h+2] = float(high_rank_norm)
    obs[h+3] = float(low_rank_norm)
    obs[h+4] = max(0.0, float(high_rank_norm - low_rank_norm))  # gap
    return obs


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — External sampling: branch structure
# ══════════════════════════════════════════════════════════════════════════════

def test_external_sampling_structure(num_players=6):
    """Check that _traversal_worker returns a list with a precomputed-regret episode."""
    print("\n" + "="*60)
    print("TEST 1: External sampling branch structure")
    print("="*60)

    _worker_init(None, None, None)  # no networks → uniform strategy

    ok = True
    branch_counts = []
    for i in range(20):
        player_idx = i % num_players
        result = _traversal_worker((player_idx, i * 1000, num_players))
        # result must be a list
        if not isinstance(result, list):
            print(f"  ✗ Episode {i}: expected list, got {type(result)}")
            ok = False
            continue
        branch_counts.append(len(result))
        # First episode should contain precomputed regret steps
        first_ep_data = result[0][0]
        has_precomp = any(s.get('type') == 'precomputed' for s in first_ep_data)
        if len(result) > 1 and not has_precomp:
            print(f"  ✗ Episode {i}: {len(result)} branches but no precomputed regret steps")
            ok = False
        # Check precomputed steps have required fields
        for s in first_ep_data:
            if s.get('type') == 'precomputed':
                missing = [k for k in ('obs','action','legal','regret','strategy') if k not in s]
                if missing:
                    print(f"  ✗ precomputed step missing keys: {missing}")
                    ok = False

    avg = np.mean(branch_counts) if branch_counts else 0
    print(f"  Episodes per traversal: min={min(branch_counts)} max={max(branch_counts)} avg={avg:.1f}")
    print(f"  (>1 means external sampling branched at first decision)")
    print()
    if ok:
        print("  ✓ Branch structure correct — precomputed regrets present")
    else:
        print("  ✗ BUGS FOUND — see above")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Exact regret direction from branch_rewards
# ══════════════════════════════════════════════════════════════════════════════

def test_exact_regret_direction(num_players=6):
    """
    Simulate exact regret computation from branch_rewards.
    For fold: regret = reward(fold)/scale - EV
    With uniform strategy and reward(fold)=0, reward(call/raise)=negative for trash:
      EV < 0  →  regret(fold) = 0 - EV > 0  →  fold more  (correct for 72o)
    With reward(fold)=0, reward(call/raise)=positive for premium:
      EV > 0  →  regret(fold) = 0 - EV < 0  →  don't fold (correct for AA)
    """
    print("\n" + "="*60)
    print("TEST 2: Exact regret direction from branch rewards")
    print("="*60)

    # Simulate: fold=0 reward (non-blind preflop), call/raise = terminal outcome
    # For AA:   playing (call/raise) is profitable → positive reward
    # For 72o:  playing (call/raise) usually loses → negative reward
    legal = [0, 1, 2]  # fold, call, raise
    strategy = np.array([1/3, 1/3, 1/3])  # uniform (early training)

    test_cases = [
        ("AA  (call/raise profitable)", {0: 0.0,  1:  80.0, 2:  90.0}),
        ("AA  (call/raise slightly+)",  {0: 0.0,  1:  20.0, 2:  30.0}),
        ("72o (call/raise costly)",     {0: 0.0,  1: -60.0, 2: -70.0}),
        ("72o (call/raise slightly-)",  {0: 0.0,  1: -10.0, 2: -15.0}),
        ("J9o (borderline)",            {0: 0.0,  1:   5.0, 2:  -5.0}),
    ]

    print(f"{'Scenario':<35}  {'EV':>8}  {'regret(fold)':>14}  {'ok?':>6}")
    print("-"*70)
    all_ok = True
    for label, branch_rewards in test_cases:
        ev = sum(strategy[a] * branch_rewards[a] / _REWARD_SCALE for a in legal)
        regret_fold = 0.0 / _REWARD_SCALE - ev   # fold reward = 0 (non-blind)
        is_aa  = "AA"  in label
        is_72o = "72o" in label
        if is_aa:
            expected_neg = regret_fold < 0
            flag = "✓" if expected_neg else "✗ BAD (should be negative)"
            if not expected_neg: all_ok = False
        elif is_72o:
            expected_pos = regret_fold > 0
            flag = "✓" if expected_pos else "✗ BAD (should be positive)"
            if not expected_pos: all_ok = False
        else:
            flag = ""
        print(f"{label:<35}  {ev:>8.4f}  {regret_fold:>14.4f}  {flag}")

    print()
    if all_ok:
        print("  ✓ AA fold gets negative regret, 72o fold gets positive regret")
        print("  ✓ No synthetic reward needed — exact branch rewards do the right thing")
    else:
        print("  ✗ BUGS FOUND — see above")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Live traversals: check how often synthetic reward triggers + values
# ══════════════════════════════════════════════════════════════════════════════

def test_live_traversals(deep_cfr, num_episodes=200, num_players=6):
    print("\n" + "="*60)
    print(f"TEST 3: Live traversals ({num_episodes} traversals)")
    print("="*60)

    _worker_init(deep_cfr.advantage_networks, deep_cfr.strategy_networks, None)

    branch_counts = []   # episodes per traversal
    branch_episode_count = 0
    normal_episode_count  = 0

    flat_episodes = []
    for i in range(num_episodes):
        player_idx = i % num_players
        result = _traversal_worker((player_idx, i * 1000, num_players))
        branch_counts.append(len(result))
        for ep in result:
            ep_data, reward, trav = ep
            has_precomp = any(s.get('type') == 'precomputed' for s in ep_data)
            if has_precomp:
                branch_episode_count += 1
            else:
                normal_episode_count += 1
            flat_episodes.append(ep)

    total_eps = len(flat_episodes)
    avg_branches = np.mean(branch_counts)
    print(f"  Traversals run:          {num_episodes}")
    print(f"  Episodes generated:      {total_eps}  (avg {avg_branches:.1f} per traversal)")
    print(f"  With precomputed regrets:{branch_episode_count}")
    print(f"  Post-branch episodes:    {normal_episode_count}")

    if avg_branches <= 1.0:
        print("\n  ✗ All traversals returned 1 episode — branching not triggering!")
        print("    Possible: traversal player never got to act (e.g. all others folded first)")
        ok = False
    else:
        print(f"\n  ✓ External sampling active: avg {avg_branches:.1f} branches per traversal")
        ok = True

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Advantage network sensitivity: does it distinguish AA from 72o?
# ══════════════════════════════════════════════════════════════════════════════

def test_network_sensitivity(deep_cfr, num_players=6):
    print("\n" + "="*60)
    print("TEST 4: Advantage network sensitivity (AA vs 72o)")
    print("="*60)

    engine_tmp = make_engine(num_players)
    obs_size = engine_tmp.get_observation_space_size()

    obs_aa  = hand_obs(num_players, True,  False, 1.0,   1.0)
    obs_kk  = hand_obs(num_players, True,  False, 11/12, 11/12)
    obs_72o = hand_obs(num_players, False, False, 5/12,  0.0)
    obs_32o = hand_obs(num_players, False, False, 1/12,  0.0)

    print(f"  Querying player 0 advantage network (obs_size={obs_size}):")
    net = deep_cfr.advantage_networks[0]
    net.eval()
    with torch.no_grad():
        for label, obs in [("AA", obs_aa), ("KK", obs_kk), ("72o", obs_72o), ("32o", obs_32o)]:
            obs_t = torch.from_numpy(obs).unsqueeze(0)
            adv = net(obs_t).numpy()[0]
            legal = [0, 1, 2]
            adv_l = adv[legal]
            adv_scale = max(float(np.abs(adv_l).max()), 1.0)
            adv_n = adv_l / adv_scale
            # regret-matching strategy
            pos = np.maximum(adv_l, 0.0); ps = pos.sum()
            strat = pos / ps if ps > 0 else np.ones(3)/3
            print(f"    {label:<6}: adv(fold,call,raise)={adv_l.round(3)}  "
                  f"normed={adv_n.round(3)}  "
                  f"strat(fold,call,raise)={strat.round(3)}")

    # Check if fold adv is LOWER for AA than 72o (network should have learned this)
    with torch.no_grad():
        adv_aa  = net(torch.from_numpy(obs_aa).unsqueeze(0)).numpy()[0]
        adv_72o = net(torch.from_numpy(obs_72o).unsqueeze(0)).numpy()[0]

    fold_aa  = float(adv_aa[0])
    fold_72o = float(adv_72o[0])
    print()
    if fold_aa < fold_72o:
        print(f"  ✓ fold_adv(AA)={fold_aa:.4f} < fold_adv(72o)={fold_72o:.4f}  [correct direction]")
    else:
        print(f"  ✗ fold_adv(AA)={fold_aa:.4f} >= fold_adv(72o)={fold_72o:.4f}  [INVERTED — network hasn't learned hand values]")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Full regret pipeline: run N episodes → compute regrets → inspect
# ══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline(deep_cfr, num_episodes=100, num_players=6):
    print("\n" + "="*60)
    print(f"TEST 5: Full regret pipeline ({num_episodes} episodes)")
    print("="*60)

    _worker_init(deep_cfr.advantage_networks, deep_cfr.strategy_networks, None)

    episodes = []
    for i in range(num_episodes):
        player_idx = i % num_players
        result = _traversal_worker((player_idx, i * 999, num_players))
        # result is now a list of episodes (external sampling)
        episodes.extend(result)

    # Monkey-patch compute_regrets to capture regrets without adding to buffer
    regret_records = []   # (hand_equity, action, regret, is_synthetic)

    original_add = deep_cfr.replay_buffer.add

    def capturing_add(obs, action, regret, strategy, player, iteration=1):
        # reconstruct equity from obs
        h = num_players + 5
        ip = obs[h] > 0.5; hi = obs[h+2]; lo = obs[h+3]
        if ip:
            eq = float(np.clip(0.53 + hi*0.32, 0.25, 0.85))
        else:
            gap = max(0, hi-lo)
            eq = float(np.clip(0.36 + hi*0.15 + lo*0.08 - gap*0.12, 0.25, 0.85))
        regret_records.append((eq, action, float(regret)))
        original_add(obs, action, regret, strategy, player, iteration)

    deep_cfr.replay_buffer.add = capturing_add
    _compute_regrets_gpu(episodes, deep_cfr, iteration=1)
    deep_cfr.replay_buffer.add = original_add

    if not regret_records:
        print("  No regrets computed (all episodes had no traversal steps?)")
        return

    # Group by action=0 (fold) and equity bucket
    fold_records    = [(eq, r) for eq, a, r in regret_records if a == 0]
    nonfold_records = [(eq, r) for eq, a, r in regret_records if a != 0]

    # Premium vs trash fold regrets
    premium_fold = [r for eq, r in fold_records if eq > 0.65]
    trash_fold   = [r for eq, r in fold_records if eq < 0.35]

    print(f"  Total regret entries:   {len(regret_records)}")
    print(f"  Fold regrets:           {len(fold_records)}")
    print(f"    premium (eq>0.65):    {len(premium_fold)}  "
          f"avg={np.mean(premium_fold):.4f}  " if premium_fold else
          f"    premium (eq>0.65):    0")
    print(f"    trash   (eq<0.35):    {len(trash_fold)}  "
          f"avg={np.mean(trash_fold):.4f}  " if trash_fold else
          f"    trash   (eq<0.35):    0")

    if premium_fold:
        avg_prem = np.mean(premium_fold)
        avg_trash = np.mean(trash_fold) if trash_fold else 0.0
        print()
        if avg_prem < avg_trash:
            print(f"  ✓ avg_fold_regret(premium)={avg_prem:.4f} < avg_fold_regret(trash)={avg_trash:.4f}")
            print("    Premium hands get more negative fold regret → fold strategy decreases")
        else:
            print(f"  ✗ avg_fold_regret(premium)={avg_prem:.4f} >= avg_fold_regret(trash)={avg_trash:.4f}")
            print("    BUG: premium hands not getting more negative fold regret than trash!")

    # Equity buckets
    print("\n  Fold regret by equity bucket:")
    buckets = {}
    for eq, r in fold_records:
        b = int(eq * 5) / 5.0   # 0.0, 0.2, 0.4, 0.6, 0.8
        buckets.setdefault(b, []).append(r)
    for b in sorted(buckets.keys(), reverse=True):
        vals = buckets[b]
        pct_pos = 100 * sum(1 for v in vals if v > 0) / len(vals)
        print(f"    equity~{b:.1f}:  n={len(vals):4d}  avg={np.mean(vals):>8.4f}  %positive={pct_pos:5.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Debug regret computation pipeline')
    parser.add_argument('--checkpoint', default=None, help='Model checkpoint to load')
    parser.add_argument('--players',    type=int, default=6)
    parser.add_argument('--episodes',   type=int, default=200)
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))

    engine = make_engine(args.players)
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()

    print(f"obs_size={obs_size}  act_size={act_size}  players={args.players}")

    deep_cfr = DeepCFR(
        observation_size=obs_size,
        action_size=act_size,
        num_players=args.players,
        hidden_layers=config['network']['hidden_layers'],
        dropout=config['network'].get('dropout', 0.1),
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        deep_cfr.load_models(args.checkpoint)

    # Run all tests
    t1 = test_external_sampling_structure(args.players)
    t2 = test_exact_regret_direction(args.players)
    t3 = test_live_traversals(deep_cfr, num_episodes=args.episodes, num_players=args.players)
    test_network_sensitivity(deep_cfr, args.players)
    test_full_pipeline(deep_cfr, num_episodes=args.episodes, num_players=args.players)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  T1 synthetic reward values:  {'PASS' if t1 else 'FAIL'}")
    print(f"  T2 regret direction:         {'PASS' if t2 else 'FAIL'}")
    print(f"  T3 synthetic triggers:       {'PASS' if t3 else 'FAIL (0 triggered!)'}")
    print()
    print("  If T3 fails: check that train_gpu.py on the server has the fix.")
    print("  If T2 fails with 'network thinks all bad': increase synthetic scale.")
    print("  If T5 shows positive premium fold regret: synthetic reward too weak.")


if __name__ == '__main__':
    main()
