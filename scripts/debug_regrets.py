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
    _preflop_fold_synthetic_reward,
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
# TEST 1 — Synthetic reward values
# ══════════════════════════════════════════════════════════════════════════════

def test_synthetic_rewards(num_players=6):
    print("\n" + "="*60)
    print("TEST 1: Synthetic reward values")
    print("="*60)

    # (label, is_pair, is_suited, high_norm, low_norm)
    # rank_norm: A=1.0, K=11/12, Q=10/12, ..., 2=0.0
    hands = [
        ("AA",      True,  False, 1.0,      1.0),
        ("KK",      True,  False, 11/12,    11/12),
        ("QQ",      True,  False, 10/12,    10/12),
        ("JJ",      True,  False, 9/12,     9/12),
        ("22",      True,  False, 0.0,      0.0),
        ("AKs",     False, True,  1.0,      11/12),
        ("AKo",     False, False, 1.0,      11/12),
        ("AQs",     False, True,  1.0,      10/12),
        ("KQo",     False, False, 11/12,    10/12),
        ("T9s",     False, True,  8/12,     7/12),
        ("72o",     False, False, 5/12,     0.0),
        ("32o",     False, False, 1/12,     0.0),
        ("trash",   False, False, 2/12,     0.0),
    ]

    baseline = 1.0 / num_players
    # Build (equity, synth) pairs and sort by equity descending for ordered check
    rows = []
    for label, is_pair, is_suited, hi, lo in hands:
        obs = hand_obs(num_players, is_pair, is_suited, hi, lo)
        synth = _preflop_fold_synthetic_reward(obs, num_players)
        if is_pair:
            equity = 0.53 + hi * 0.32
        else:
            gap = max(0.0, hi - lo)
            equity = 0.36 + hi*0.15 + lo*0.08 + (0.03 if is_suited else 0) - gap*0.12
        equity = float(np.clip(equity, 0.25, 0.85))
        rows.append((label, equity, synth))

    # Sort by equity descending for the monotone check
    rows.sort(key=lambda x: -x[1])

    print(f"{'Hand':<12} {'equity':>8} {'synth_reward':>14} {'regret(fold,ev=0)':>18}")
    print("-"*55)
    ok = True
    prev_equity, prev_synth = None, None
    for label, equity, synth in rows:
        regret_at_ev0 = synth - 0.0
        flag = ""
        # A weaker hand should never get MORE negative synth than a stronger hand
        if prev_synth is not None and prev_equity is not None:
            if equity < prev_equity and synth < prev_synth - 0.01:
                flag = "  ← BUG: weaker hand has MORE negative synth"
                ok = False
        print(f"{label:<12} {equity:>8.3f} {synth:>14.4f} {regret_at_ev0:>18.4f}{flag}")
        prev_equity, prev_synth = equity, synth

    print()
    if ok:
        print("  ✓ Synthetic rewards: strongest hands most negative, weak hands near 0")
    else:
        print("  ✗ BUGS FOUND — see above")
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Regret direction with mock advantages
# ══════════════════════════════════════════════════════════════════════════════

def test_regret_direction(num_players=6):
    """
    Simulate _compute_regrets_gpu logic for a single fold episode.
    Test with various adv network outputs to see when fold still gets
    positive regret despite synthetic reward.
    """
    print("\n" + "="*60)
    print("TEST 2: Regret direction (synthetic reward vs network EV)")
    print("="*60)

    obs_aa   = hand_obs(num_players, True,  False, 1.0,   1.0)
    obs_72o  = hand_obs(num_players, False, False, 5/12,  0.0)

    synth_aa  = _preflop_fold_synthetic_reward(obs_aa,  num_players)
    synth_72o = _preflop_fold_synthetic_reward(obs_72o, num_players)

    print(f"  synth(AA)  = {synth_aa:.4f}")
    print(f"  synth(72o) = {synth_72o:.4f}")
    print()

    # Simulate for different network EV estimates
    # adv = [fold_score, call_score, raise_score] (5 actions but only 3 legal preflop)
    legal = [0, 1, 2]  # fold=0, call=1, raise=2
    action = 0         # fold was sampled

    test_cases = [
        ("random init (all zero)",    [0.0,  0.0,  0.0]),
        ("random init (noisy)",       [-0.1, 0.1,  0.2]),
        ("network thinks all bad",    [-0.5, -0.3, -0.4]),
        ("network thinks play=bad",   [-0.1, -0.8, -0.9]),
        ("network thinks play=great", [-0.1,  0.8,  0.9]),
    ]

    print(f"{'Scenario':<35} {'EV':>6}  {'regret(fold,AA)':>16}  {'regret(fold,72o)':>16}  {'AA_ok?':>6}")
    print("-"*85)
    all_ok = True
    for label, adv_raw in test_cases:
        for hand_label, obs, synth in [("AA", obs_aa, synth_aa), ("72o", obs_72o, synth_72o)]:
            adv = np.array(adv_raw[:5] if len(adv_raw)>=5 else adv_raw + [0.0]*(5-len(adv_raw)), dtype=np.float32)
            mask = np.zeros(5, dtype=np.float32); mask[legal] = 1.0
            adv_m = adv * mask

            pos = np.maximum(adv_m, 0.0) * mask
            ps  = pos.sum()
            strat_init = pos / ps if ps > 0 else mask / mask.sum()

            legal_adv = adv_m[legal]
            adv_scale = max(float(np.abs(legal_adv).max()), 1.0)
            adv_normed = adv_m / adv_scale

            ev = float(np.sum(strat_init[legal] * adv_normed[legal]))
            regret_fold = synth - ev  # action == 0 (fold) was sampled

        # Redo for both hands
        results = {}
        for hl, obs_h, sy in [("AA", obs_aa, synth_aa), ("72o", obs_72o, synth_72o)]:
            adv = np.array(adv_raw + [0.0]*(5-len(adv_raw)), dtype=np.float32)
            mask = np.zeros(5, dtype=np.float32); mask[legal] = 1.0
            adv_m = adv * mask
            pos = np.maximum(adv_m, 0.0) * mask; ps = pos.sum()
            strat_init = pos / ps if ps > 0 else mask / mask.sum()
            legal_adv = adv_m[legal]
            adv_scale = max(float(np.abs(legal_adv).max()), 1.0)
            adv_normed = adv_m / adv_scale
            ev = float(np.sum(strat_init[legal] * adv_normed[legal]))
            results[hl] = (ev, sy - ev)

        ev_aa, r_aa   = results["AA"]
        ev_72, r_72   = results["72o"]
        aa_ok = r_aa < 0   # fold AA should get negative regret
        flag = "✓" if aa_ok else "✗ BAD"
        if not aa_ok:
            all_ok = False
        print(f"{label:<35} {ev_aa:>6.3f}  {r_aa:>16.4f}  {r_72:>16.4f}  {flag:>6}")

    print()
    if all_ok:
        print("  ✓ In all cases, fold AA gets negative regret")
    else:
        print("  ✗ Some cases give fold AA positive regret — synthetic reward too weak!")
    return all_ok


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Live traversals: check how often synthetic reward triggers + values
# ══════════════════════════════════════════════════════════════════════════════

def test_live_traversals(deep_cfr, num_episodes=200, num_players=6):
    print("\n" + "="*60)
    print(f"TEST 3: Live traversals ({num_episodes} episodes)")
    print("="*60)

    engine_tmp = make_engine(num_players)
    obs_size = engine_tmp.get_observation_space_size()

    # Init worker globals
    _worker_init(deep_cfr.advantage_networks, deep_cfr.strategy_networks, None)

    synthetic_count = 0
    regular_count   = 0
    synth_by_hand_label = {}  # equity_bucket → [synth_rewards]
    fold_regrets_high = []    # equity > 0.6 → premium hand
    fold_regrets_low  = []    # equity < 0.35 → trash hand

    all_episodes = []
    for i in range(num_episodes):
        player_idx = i % num_players
        result = _traversal_worker((player_idx, i * 1000, num_players))
        all_episodes.append(result)

    # Analyse raw rewards
    for ep_data, raw_reward, trav_player in all_episodes:
        trav_steps = [s for s in ep_data if s['player'] == trav_player]
        is_synthetic = (
            abs(raw_reward) < 0.5 and
            len(trav_steps) == 1 and
            trav_steps[0]['action'] == 0
        )
        if is_synthetic:
            synthetic_count += 1
            obs  = trav_steps[0]['obs']
            synth = _preflop_fold_synthetic_reward(obs, num_players)
            # bucket by equity
            h = num_players + 5
            ip = obs[h] > 0.5
            hi = obs[h+2]; lo = obs[h+3]
            if ip:
                eq = 0.53 + hi * 0.32
            else:
                gap = max(0, hi - lo)
                eq = 0.36 + hi*0.15 + lo*0.08 - gap*0.12
            eq = float(np.clip(eq, 0.25, 0.85))
            bucket = f"eq={eq:.2f}"
            synth_by_hand_label.setdefault(bucket, []).append(synth)
        else:
            regular_count += 1

    total = synthetic_count + regular_count
    print(f"  Episodes:          {total}")
    print(f"  Synthetic reward:  {synthetic_count} ({100*synthetic_count/total:.1f}%)")
    print(f"  Regular reward:    {regular_count} ({100*regular_count/total:.1f}%)")

    if synthetic_count == 0:
        print("\n  ✗ ZERO synthetic rewards triggered!")
        print("    Possible causes:")
        print("    - All traversals start from SB/BB (always post chips)")
        print("    - Game engine assigns non-zero rewards to preflop folders")
        print("    - The fix is not running (wrong file on server)")
    else:
        print("\n  Synthetic reward distribution (sample of equity buckets):")
        # sort by equity descending
        for bucket, vals in sorted(synth_by_hand_label.items(), reverse=True)[:10]:
            avg = np.mean(vals)
            cnt = len(vals)
            print(f"    {bucket:>12}  avg_synth={avg:>8.4f}  count={cnt}")

    return synthetic_count > 0


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
        episodes.append(result)

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
    t1 = test_synthetic_rewards(args.players)
    t2 = test_regret_direction(args.players)
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
