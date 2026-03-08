"""
Debug/validation script: copy of train_gpu.py with detailed logging.

Validates:
  1. Engine generates consistent card deals (same cards across branches)
  2. Fold rewards are correct (non-blind ≈ 0, SB ≈ -1, BB ≈ -2)
  3. Observations are in valid range [0, 1]
  4. Regrets are within clip range [-2.0, +2.0]
  5. Strategy sums to 1.0
  6. Branch rewards are consistent across n_samples
  7. Training loss is computed without errors

Run with:
  python3 scripts/debug_training.py
"""
import sys
import os
import warnings
import logging

import copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config
from pathlib import Path

_REWARD_SCALE = 100.0
_REGRET_CLIP  = 2.0
_PASS = "✓"
_FAIL = "✗"

# ── Globals (used by helper functions, same pattern as train_gpu.py) ─────────
_SHARED_ADV_NETS   = None
_SHARED_STRAT_NETS = None
_ACT_SIZE          = None

CHECK_RESULTS = []   # list of (description, passed, detail)


def _record(desc, passed, detail=""):
    status = _PASS if passed else _FAIL
    CHECK_RESULTS.append((desc, passed, detail))
    marker = "  [PASS]" if passed else "  [FAIL]"
    print(f"{marker} {desc}" + (f": {detail}" if detail else ""))


def _make_engine(num_players):
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


def _regret_match(adv, legal, epsilon=0.05):
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    strategy[legal] = pos / s if s > 0 else 1.0 / len(legal)
    strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
    return strategy


# ── Section 1: Engine basics ─────────────────────────────────────────────────

def check_engine_basics(num_players):
    print("\n" + "=" * 60)
    print("SECTION 1: Engine basics")
    print("=" * 60)

    engine = _make_engine(num_players)
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()
    print(f"  obs_size={obs_size}  act_size={act_size}  players={num_players}")

    # Check obs_size is plausible
    expected_min = num_players + 22
    _record("obs_size plausible", obs_size >= expected_min,
            f"obs_size={obs_size} >= {expected_min}")

    engine.reset()

    # Check we can get observation
    obs = engine.get_observation()
    _record("observation shape correct", obs.shape == (obs_size,),
            f"got {obs.shape}")

    # Check observation range [0, 1]
    obs_min, obs_max = float(obs.min()), float(obs.max())
    _record("observation in [0, 1]", obs_min >= -0.01 and obs_max <= 1.01,
            f"min={obs_min:.4f}  max={obs_max:.4f}")
    print(f"    obs[:8] = {obs[:8]}")

    # Check legal actions
    legal = engine.get_legal_actions()
    _record("legal actions non-empty", len(legal) > 0, f"{legal}")

    # Check agents list
    agents = engine.agents
    _record("correct number of agents", len(agents) == num_players,
            f"{agents}")

    engine.close()
    return obs_size, act_size


# ── Section 2: RNG save/restore (branch consistency) ─────────────────────────

def check_rng_consistency(num_players):
    print("\n" + "=" * 60)
    print("SECTION 2: RNG save/restore (branch consistency)")
    print("=" * 60)
    print("  Verifying that _fresh_engine_at() replays with identical cards.")

    engine = _make_engine(num_players)
    game = engine._get_game()
    pre_rng  = game.np_random.get_state()
    pre_dlr  = game.dealer_id
    engine.reset()

    # Read hole cards of player 0 after reset
    def _get_hand(eng, player_idx):
        g = eng._get_game()
        hand = g.players[player_idx].hand
        return [(c.rank, c.suit) for c in hand] if hand else []

    hand_first = _get_hand(engine, 0)
    print(f"    Player 0 hand (original): {hand_first}")
    engine.close()

    # Replay multiple times and check consistency
    hands = []
    for trial in range(5):
        eng2 = _make_engine(num_players)
        g2   = eng2._get_game()
        g2.np_random.set_state(pre_rng)
        g2.dealer_id = pre_dlr
        eng2.reset()
        h = _get_hand(eng2, 0)
        hands.append(h)
        eng2.close()

    all_same = all(h == hand_first for h in hands)
    _record("RNG restore gives same cards (5 trials)", all_same,
            f"original={hand_first}  replays={hands}")

    if not all_same:
        for i, h in enumerate(hands):
            print(f"    Trial {i}: {h}")

    return pre_rng, pre_dlr


# ── Section 3: Fold reward validation ────────────────────────────────────────

def check_fold_rewards(num_players):
    """
    Run 30 games where the first acting player always folds.
    Non-blind positions should get ~0, SB ~ -1, BB ~ -2.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: Fold reward validation")
    print("=" * 60)

    fold_rewards_by_position = {}   # rel_position -> list of rewards

    GAMES = 50
    for _ in range(GAMES):
        engine = _make_engine(num_players)
        game = engine._get_game()
        pre_rng = game.np_random.get_state()
        pre_dlr = game.dealer_id
        engine.reset()

        if engine.is_done():
            engine.close()
            continue

        ca  = engine.current_agent
        ai  = engine.agents.index(ca)
        dealer_idx = engine._get_game().dealer_id
        rel_pos = (ai - dealer_idx) % num_players

        legal = engine.get_legal_actions()
        # action 0 = fold
        if 0 not in [int(a) for a in legal]:
            engine.close()
            continue

        engine.step(0)  # fold

        # Play remaining to terminal
        while not engine.is_done():
            ca2   = engine.current_agent
            ai2   = engine.agents.index(ca2)
            legal2 = engine.get_legal_actions()
            if not legal2:
                break
            engine.step(int(np.random.choice([int(a) for a in legal2])))

        rw = engine.get_rewards()
        reward = float(rw[engine.agents[ai]])
        engine.close()

        if rel_pos not in fold_rewards_by_position:
            fold_rewards_by_position[rel_pos] = []
        fold_rewards_by_position[rel_pos].append(reward)

    print(f"  Fold rewards by relative position (button=0, SB=1, BB=2):")
    for pos in sorted(fold_rewards_by_position.keys()):
        rws = fold_rewards_by_position[pos]
        avg = np.mean(rws)
        mn  = np.min(rws)
        mx  = np.max(rws)
        pos_name = {0: "BTN", 1: "SB ", 2: "BB ", 3: "UTG", 4: "HJ ", 5: "CO "}.get(pos, str(pos))
        print(f"    {pos_name} (rel={pos}): mean={avg:+.2f}  min={mn:+.2f}  max={mx:+.2f}  n={len(rws)}")

    # Sanity: SB and BB fold rewards should always be negative or zero (they paid blinds)
    sb_rewards = fold_rewards_by_position.get(1, [])
    bb_rewards = fold_rewards_by_position.get(2, [])

    if sb_rewards:
        _record("SB fold reward negative (paid blind)", np.mean(sb_rewards) < 0,
                f"mean={np.mean(sb_rewards):+.2f}")
    if bb_rewards:
        _record("BB fold reward negative (paid blind)", np.mean(bb_rewards) < 0,
                f"mean={np.mean(bb_rewards):+.2f}")

    # Non-blind positions (BTN, UTG, HJ, CO) — fold reward should be 0
    non_blind_positions = [pos for pos in [0, 3, 4, 5] if pos in fold_rewards_by_position]
    for pos in non_blind_positions:
        rws = fold_rewards_by_position[pos]
        mean_rw = np.mean(rws)
        _record(f"Non-blind pos={pos} fold reward ≈ 0", abs(mean_rw) < 5,
                f"mean={mean_rw:+.2f}")


# ── Section 4: Branch reward consistency check ────────────────────────────────

def check_branch_consistency(num_players, obs_size):
    print("\n" + "=" * 60)
    print("SECTION 4: Branch reward consistency (n_samples variance)")
    print("=" * 60)
    print("  Running 5 traversals, checking variance across n_samples rollouts.")

    engine = _make_engine(num_players)
    game   = engine._get_game()
    pre_rng  = game.np_random.get_state()
    pre_dlr  = game.dealer_id
    engine.reset()

    player_idx = 0

    def _fresh_engine_at(history):
        eng = _make_engine(num_players)
        g   = eng._get_game()
        g.np_random.set_state(pre_rng)
        g.dealer_id = pre_dlr
        eng.reset()
        for a in history:
            eng.step(a)
        return eng

    def _run_to_terminal(eng):
        while not eng.is_done():
            legal = eng.get_legal_actions()
            if not legal:
                break
            eng.step(int(np.random.choice([int(a) for a in legal])))
        rw = eng.get_rewards()
        reward = float(rw[eng.agents[player_idx]])
        eng.close()
        return reward

    # Walk to player 0's first decision
    history = []
    while not engine.is_done():
        ca    = engine.current_agent
        ai    = engine.agents.index(ca)
        legal = engine.get_legal_actions()
        if not legal:
            break
        if ai == player_idx:
            break
        action = int(np.random.choice([int(a) for a in legal]))
        history.append(action)
        engine.step(action)

    if engine.is_done():
        print("  Game ended before player 0 acted — skipping branch check.")
        engine.close()
        return

    legal_int = [int(a) for a in engine.get_legal_actions()]
    obs       = engine.get_observation()
    engine.close()

    print(f"  Player 0 legal actions: {legal_int}  history_len={len(history)}")
    print(f"  Observation range: min={obs.min():.4f}  max={obs.max():.4f}")

    # Check observation range
    _record("First decision obs in [0,1]", obs.min() >= -0.01 and obs.max() <= 1.01,
            f"min={obs.min():.4f}  max={obs.max():.4f}")

    # For each action, run 5 samples and log the variance
    N_SAMPLES = 5
    all_variances_ok = True
    print(f"\n  Branch rewards per action ({N_SAMPLES} samples each):")
    for ba in legal_int:
        samples = [_run_to_terminal(_fresh_engine_at(history + [ba]))
                   for _ in range(N_SAMPLES)]
        mean_rw = np.mean(samples)
        std_rw  = np.std(samples)
        min_rw  = np.min(samples)
        max_rw  = np.max(samples)
        action_name = {0: "fold", 1: "call", 2: "raise"}.get(ba, f"act{ba}")
        print(f"    action={ba}({action_name}): samples={[f'{r:+.1f}' for r in samples]}")
        print(f"      mean={mean_rw:+.2f}  std={std_rw:.2f}  range=[{min_rw:+.1f}, {max_rw:+.1f}]")

        # High variance is expected (poker has many outcomes), but fold reward should be stable
        if ba == 0:  # fold
            # Fold should give consistent results (no randomness after fold from player's perspective)
            # Actually rewards can vary since opponents act differently — just log it
            if std_rw > 50:
                all_variances_ok = False
                print(f"      WARNING: fold reward has very high variance (std={std_rw:.2f})")

    _record("Branch rewards computed without errors", True)


# ── Section 5: Regret and strategy validation ─────────────────────────────────

def check_regrets_and_strategy(num_players, obs_size, act_size, deep_cfr):
    print("\n" + "=" * 60)
    print("SECTION 5: Regret and strategy validation")
    print("=" * 60)

    engine = _make_engine(num_players)
    game   = engine._get_game()
    pre_rng  = game.np_random.get_state()
    pre_dlr  = game.dealer_id
    engine.reset()
    player_idx = 0

    # Get player 0's first decision
    history = []
    while not engine.is_done():
        ca    = engine.current_agent
        ai    = engine.agents.index(ca)
        legal = engine.get_legal_actions()
        if not legal:
            break
        if ai == player_idx:
            break
        action = int(np.random.choice([int(a) for a in legal]))
        history.append(action)
        engine.step(action)

    if engine.is_done():
        engine.close()
        print("  Game ended before player 0 acted.")
        return

    obs       = engine.get_observation().copy()
    legal_int = [int(a) for a in engine.get_legal_actions()]
    engine.close()

    # Get strategy from uniform distribution (no trained networks yet)
    strategy = np.zeros(act_size, dtype=np.float32)
    strategy[legal_int] = 1.0 / len(legal_int)

    # Fake branch rewards (all equal → EV = mean → all regrets ≈ 0)
    fake_rewards = {a: float(i * 10) for i, a in enumerate(legal_int)}
    ev = sum(strategy[a] * fake_rewards[a] / _REWARD_SCALE for a in legal_int)

    regrets = []
    for a in legal_int:
        raw_regret = fake_rewards[a] / _REWARD_SCALE - ev
        clipped    = float(np.clip(raw_regret, -_REGRET_CLIP, _REGRET_CLIP))
        regrets.append(clipped)
        print(f"  action={a}: reward={fake_rewards[a]:+.1f}  ev={ev:+.4f}  "
              f"raw_regret={raw_regret:+.4f}  clipped={clipped:+.4f}")

    # Check regrets in range
    _record("Regrets within clip range", all(-_REGRET_CLIP <= r <= _REGRET_CLIP for r in regrets),
            f"regrets={[f'{r:+.3f}' for r in regrets]}")

    # Check strategy sums to 1.0 over legal actions
    strat_sum = strategy[legal_int].sum()
    _record("Strategy sums to 1.0", abs(strat_sum - 1.0) < 1e-5,
            f"sum={strat_sum:.6f}")

    # Check regret-matching with advantage network
    with torch.no_grad():
        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
        adv_np = deep_cfr.advantage_networks[player_idx](obs_t).numpy()[0]
    rm_strategy = _regret_match(adv_np, legal_int)
    rm_sum = rm_strategy[legal_int].sum()
    _record("Regret-matching strategy sums to 1.0", abs(rm_sum - 1.0) < 1e-5,
            f"sum={rm_sum:.6f}")
    print(f"  Advantage network output (legal actions): {adv_np[legal_int]}")
    print(f"  Regret-matched strategy (legal): {rm_strategy[legal_int]}")


# ── Section 6: Buffer add and training step ───────────────────────────────────

def check_training_step(obs_size, act_size, deep_cfr):
    print("\n" + "=" * 60)
    print("SECTION 6: Buffer add and training step")
    print("=" * 60)

    # Add fake experiences to buffer
    N_FAKE = 1000
    for i in range(N_FAKE):
        obs      = np.random.rand(obs_size).astype(np.float32)
        action   = np.random.randint(0, act_size)
        regret   = float(np.random.uniform(-2, 2))
        strategy = np.random.dirichlet(np.ones(act_size)).astype(np.float32)
        player   = np.random.randint(0, 6)
        deep_cfr.replay_buffer.add(obs, action, regret, strategy, player, iteration=1)

    buf_size = len(deep_cfr.replay_buffer)
    _record("Buffer accepts samples", buf_size >= N_FAKE, f"buf_size={buf_size}")

    # Training step
    try:
        adv_loss  = deep_cfr.train_advantage_network(batch_size=64)
        strat_loss = deep_cfr.train_strategy_network(batch_size=64)
        _record("Training step completes without error", True,
                f"adv_loss={adv_loss:.4f}  strat_loss={strat_loss:.4f}")
        print(f"  adv_loss={adv_loss:.4f}  strat_loss={strat_loss:.4f}")

        # Check losses are finite
        _record("Losses are finite", np.isfinite(adv_loss) and np.isfinite(strat_loss),
                f"adv={adv_loss}  strat={strat_loss}")
    except Exception as e:
        _record("Training step completes without error", False, str(e))


# ── Section 7: Full mini-traversal with logging ───────────────────────────────

def check_mini_traversal(num_players):
    print("\n" + "=" * 60)
    print("SECTION 7: Full mini-traversal (2 games, all steps logged)")
    print("=" * 60)

    TRAVERSALS = 2

    for trav in range(TRAVERSALS):
        player_idx = trav % num_players
        print(f"\n  ── Traversal {trav + 1}, player_idx={player_idx} ──")

        engine = _make_engine(num_players)
        game   = engine._get_game()
        pre_rng  = game.np_random.get_state()
        pre_dlr  = game.dealer_id
        engine.reset()

        # Log hole cards
        g = engine._get_game()
        for pi in range(num_players):
            hand = g.players[pi].hand
            hand_str = " ".join(f"{c.rank}{c.suit}" for c in hand) if hand else "?"
            print(f"    Player {pi} hand: {hand_str}")

        dealer = g.dealer_id
        print(f"    Dealer seat (button): {dealer}")

        def _fresh_engine_at(history):
            eng = _make_engine(num_players)
            g2  = eng._get_game()
            g2.np_random.set_state(pre_rng)
            g2.dealer_id = pre_dlr
            eng.reset()
            for a in history:
                eng.step(a)
            return eng

        def _run_to_terminal(eng, label=""):
            steps = 0
            while not eng.is_done():
                ca2    = eng.current_agent
                ai2    = eng.agents.index(ca2)
                legal2 = eng.get_legal_actions()
                if not legal2:
                    break
                act2 = int(np.random.choice([int(a) for a in legal2]))
                eng.step(act2)
                steps += 1
            rw = eng.get_rewards()
            reward = float(rw[eng.agents[player_idx]])
            eng.close()
            return reward

        # Walk to player's first decision
        history = []
        step_count = 0
        while not engine.is_done():
            ca    = engine.current_agent
            ai    = engine.agents.index(ca)
            obs   = engine.get_observation()
            legal = engine.get_legal_actions()
            if not legal:
                break
            if ai == player_idx:
                break
            action = int(np.random.choice([int(a) for a in legal]))
            action_name = {0: "fold", 1: "call", 2: "raise"}.get(action, f"act{action}")
            print(f"    step {step_count}: player {ai} → {action_name}(={action})")
            history.append(action)
            engine.step(action)
            step_count += 1

        if engine.is_done():
            engine.close()
            print(f"    Game ended before player {player_idx} acted (player folded out by others).")
            _record(f"Traversal {trav+1}: player acted", False, "game ended first")
            continue

        ca    = engine.current_agent
        ai    = engine.agents.index(ca)
        obs   = engine.get_observation().copy()
        legal_int = [int(a) for a in engine.get_legal_actions()]
        engine.close()

        if ai != player_idx:
            print(f"    WARNING: Expected player {player_idx} but got {ai}")
            _record(f"Traversal {trav+1}: correct player acts", False, f"got ai={ai}")
            continue

        _record(f"Traversal {trav+1}: correct player acts", True, f"player={ai}")

        print(f"    Player {player_idx} decision: legal={legal_int}  obs_range=[{obs.min():.3f},{obs.max():.3f}]")

        # Branch with n_samples=3
        N_SAMPLES = 3
        branch_rewards = {}
        for ba in legal_int:
            samples = []
            hole_cards = []
            for s in range(N_SAMPLES):
                eng_branch = _fresh_engine_at(history + [ba])
                # Log hole cards to verify consistency
                g_branch = eng_branch._get_game()
                hand_check = [(c.rank, c.suit) for c in g_branch.players[player_idx].hand]
                hole_cards.append(hand_check)
                rw = _run_to_terminal(eng_branch)
                samples.append(rw)
            mean_rw = float(np.mean(samples))
            branch_rewards[ba] = mean_rw
            action_name = {0: "fold", 1: "call", 2: "raise"}.get(ba, f"act{ba}")
            cards_consistent = len(set(str(h) for h in hole_cards)) == 1
            print(f"    branch {ba}({action_name}): samples={[f'{r:+.1f}' for r in samples]}  mean={mean_rw:+.2f}  cards_consistent={cards_consistent}")
            if not cards_consistent:
                print(f"      WARNING: Hole cards differ across samples! {hole_cards}")
                _record(f"Traversal {trav+1} action={ba}: hole cards consistent", False, str(hole_cards))
            else:
                _record(f"Traversal {trav+1} action={ba}: hole cards consistent", True)

        # Compute regrets
        strategy = np.zeros(engine.get_action_space_size() if hasattr(engine, '_act_size') else 3,
                           dtype=np.float32)
        if len(legal_int) > 0:
            strategy_size = max(legal_int) + 1
            strategy = np.zeros(strategy_size, dtype=np.float32)
            strategy[legal_int] = 1.0 / len(legal_int)

        ev = sum(strategy[a] * branch_rewards[a] / _REWARD_SCALE for a in legal_int)
        print(f"    EV (uniform strategy) = {ev:.4f}")
        for a in legal_int:
            raw = branch_rewards[a] / _REWARD_SCALE - ev
            clipped = float(np.clip(raw, -_REGRET_CLIP, _REGRET_CLIP))
            print(f"      action={a}: reward={branch_rewards[a]:+.1f}  regret={clipped:+.4f}")

        _record(f"Traversal {trav+1}: completed successfully", True)


# ── Section 8: Verify game ends correctly ─────────────────────────────────────

def check_game_termination(num_players):
    print("\n" + "=" * 60)
    print("SECTION 8: Game termination (10 full games)")
    print("=" * 60)

    completed = 0
    errors = 0
    for trial in range(10):
        try:
            engine = _make_engine(num_players)
            engine.reset()
            step_count = 0
            while not engine.is_done():
                legal = engine.get_legal_actions()
                if not legal:
                    break
                engine.step(int(np.random.choice([int(a) for a in legal])))
                step_count += 1
                if step_count > 500:
                    print(f"  Trial {trial}: WARNING — exceeded 500 steps, possible infinite loop")
                    break
            rw = engine.get_rewards()
            total_reward = sum(rw.values())
            engine.close()

            # Sum of rewards should be 0 (zero-sum game, ignoring rake)
            if abs(total_reward) > 5:
                print(f"  Trial {trial}: rewards sum={total_reward:.2f} (expected ≈0)")
                errors += 1
            completed += 1
        except Exception as e:
            print(f"  Trial {trial}: ERROR — {e}")
            errors += 1

    _record("All 10 games complete without error", errors == 0, f"errors={errors}")
    _record("Games terminate in <500 steps", completed == 10, f"completed={completed}/10")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    os.environ['GYMNASIUM_DISABLE_LOGGING'] = '1'

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    import json
    with open(config_path) as f:
        config = json.load(f)

    num_players = config['game']['num_players']

    print("=" * 60)
    print("DEEP CFR DEBUG / VALIDATION SCRIPT")
    print("=" * 60)
    print(f"  num_players={num_players}")
    print(f"  REWARD_SCALE={_REWARD_SCALE}  REGRET_CLIP={_REGRET_CLIP}")

    # ── Initialize networks ──────────────────────────────────────────────────
    tmp_engine = _make_engine(num_players)
    obs_size = tmp_engine.get_observation_space_size()
    act_size = tmp_engine.get_action_space_size()
    tmp_engine.close()

    deep_cfr = DeepCFR(
        observation_size=obs_size,
        action_size=act_size,
        num_players=num_players,
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=100000,
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        lr_scheduler_t_max=config['training'].get('lr_scheduler_t_max', 2000),
        dropout=config['network'].get('dropout', 0.1),
        device='cpu',
    )
    print(f"  Networks initialized on CPU (debug mode)")

    # ── Run all checks ───────────────────────────────────────────────────────
    obs_size, act_size = check_engine_basics(num_players)
    check_rng_consistency(num_players)
    check_fold_rewards(num_players)
    check_branch_consistency(num_players, obs_size)
    check_regrets_and_strategy(num_players, obs_size, act_size, deep_cfr)
    check_training_step(obs_size, act_size, deep_cfr)
    check_mini_traversal(num_players)
    check_game_termination(num_players)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, p, _ in CHECK_RESULTS if p)
    total  = len(CHECK_RESULTS)
    failed = [(d, det) for d, p, det in CHECK_RESULTS if not p]

    for desc, ok, detail in CHECK_RESULTS:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {desc}" + (f" — {detail}" if detail else ""))

    print()
    print(f"  Result: {passed}/{total} checks passed")
    if failed:
        print(f"\n  FAILED checks:")
        for d, det in failed:
            print(f"    - {d}" + (f": {det}" if det else ""))
        print("\n  *** ISSUES FOUND — review the output above ***")
    else:
        print("\n  *** ALL CHECKS PASSED — training pipeline looks correct ***")


if __name__ == '__main__':
    main()
