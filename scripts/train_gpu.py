"""
Deep CFR training: parallel CPU traversals + batched GPU training.

Workers run full external-sampling MCCFR traversals covering all streets.
At every traversal player decision (preflop and postflop), all legal actions
are branched with exact terminal rewards — no network bootstrapping anywhere.
Main process handles all GPU network inference and training.
"""
import sys
import os
import warnings
import logging
import argparse
import time
import json
import multiprocessing as mp
from multiprocessing import Pool
from pathlib import Path
from collections import deque

import copy
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from torch.utils.tensorboard import SummaryWriter

from src.training import DeepCFR
from src.utils import load_config
from src.archetypes import (
    get_archetype_skew, get_archetype_profile, random_archetype,
    OPPONENT_PROFILE_DIM,
)
from validate_spots import run_validation, validation_summary_line


# Shared-memory CPU networks — set once per worker, updated by main process each iteration.
_SHARED_ADV_NETS   = None
_SHARED_STRAT_NETS = None
_ACT_SIZE          = None   # action space size, needed for uniform fallback strategy

# K-Best Self-Play: pool of historical strategy networks for diverse opponents
_KBEST_POOL = None  # list of (label, [net_per_player]) or None if disabled

_REWARD_SCALE = 100.0
_REGRET_CLIP  = 1.0
_RAKE_PCT     = 0.0
_RAKE_CAP     = 0.0

# Worker-local engine pool — reuse PettingZoo envs instead of recreating per branch
_ENGINE_POOL = []

# Preflop range override — loaded once per worker
_PREFLOP_RANGES = None      # PreflopRanges instance (or None if disabled)

# Archetype opponent training — set per worker via _worker_init
_ARCHETYPE_MODE = False


def _get_engine(num_players):
    """Get engine from pool or create new one."""
    global _ENGINE_POOL
    if _ENGINE_POOL:
        return _ENGINE_POOL.pop()
    return _make_engine(num_players)


def _return_engine(eng):
    """Return engine to pool for reuse (avoids env teardown + recreation)."""
    _ENGINE_POOL.append(eng)


def _worker_init(shared_adv, shared_strat, act_size, use_preflop_ranges=False,
                 archetype_mode=False, kbest_pool=None):
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE, _PREFLOP_RANGES
    global _ARCHETYPE_MODE, _KBEST_POOL
    _SHARED_ADV_NETS  = shared_adv
    _SHARED_STRAT_NETS = shared_strat
    _ACT_SIZE          = act_size
    _ARCHETYPE_MODE    = archetype_mode
    _KBEST_POOL        = kbest_pool
    if use_preflop_ranges:
        from src.preflop_ranges import PreflopRanges
        _PREFLOP_RANGES = PreflopRanges()
    else:
        _PREFLOP_RANGES = None
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    os.environ['GYMNASIUM_DISABLE_LOGGING'] = '1'
    # Permanently suppress C-level stdout in workers to silence Rust pokers
    # library debug prints (Winner id / Ranks on showdown). Workers return
    # results via IPC, so stdout is not needed.
    sys.stdout.flush()
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.close(devnull)


def _regret_match(adv, legal, epsilon=0.05):
    """Regret-matching with epsilon-uniform exploration and action floor.

    The action floor prevents any legal action from dropping to 0% probability.
    This is critical for call (action 1) which tends to collapse to 0% in
    self-play, creating the raise-or-fold degenerate equilibrium.
    Standard epsilon exploration is uniform across all actions.
    The action floor specifically ensures call retains minimum probability.
    """
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    strategy[legal] = pos / s if s > 0 else 1.0 / len(legal)
    if epsilon > 0:
        strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
    # Action floor: ensure call (action 1) never drops below 3%
    # This prevents the call-collapse that causes strategy oscillation.
    # 3% is small enough to not distort Nash convergence but large enough
    # to keep call regrets from going permanently negative.
    CALL = 1
    CALL_FLOOR = 0.03
    if CALL in legal and strategy[CALL] < CALL_FLOOR:
        deficit = CALL_FLOOR - strategy[CALL]
        strategy[CALL] = CALL_FLOOR
        # Take from the largest non-call action
        others = [a for a in legal if a != CALL]
        if others:
            biggest = max(others, key=lambda a: strategy[a])
            strategy[biggest] = max(0.0, strategy[biggest] - deficit)
            # Renormalize
            s = strategy[legal].sum()
            if s > 0:
                strategy[legal] /= s
    return strategy


def _make_engine(num_players):
    from src.engine import EnhancedPokerEngine
    return EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
        add_betting_history_features=True,
        behavioral_stats=True,
        opponent_profile=_ARCHETYPE_MODE,
        rake_pct=_RAKE_PCT,
        rake_cap=_RAKE_CAP,
    )


def _get_preflop_info(eng, player_idx, num_players):
    """Extract hole cards, relative position, and pot in BB for preflop override."""
    agent = eng.agents[player_idx]
    cards = eng.get_player_cards(agent)  # e.g. ['As', 'Kh']

    # Relative position
    button_idx = eng._button
    rel_pos = (player_idx - button_idx) % num_players

    # Pot in BB (BB = 2 chips)
    pot_bb = eng.get_pot_size() / 2.0

    return cards, rel_pos, pot_bb


def _traversal_worker(args):
    """
    External-sampling MCCFR with exact regrets at every traversal player decision.

    At EVERY decision point (preflop AND postflop):
      - Branch all legal actions, play each to terminal → exact rewards
      - Compute exact regrets (no network bootstrapping)
      - Sample one action from regret-matching strategy, continue

    All branches replay the full action history from game start using the
    saved game.np_random + dealer_id for identical card deals every time.

    If _PREFLOP_RANGES is set, preflop decisions use range charts instead
    of the neural network. Regrets are still computed and stored so the
    advantage network learns the correct preflop values over time.
    """
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE, _PREFLOP_RANGES
    global _ARCHETYPE_MODE, _KBEST_POOL
    player_idx, num_players, current_iter = args

    # ── Three-phase training ─────────────────────────────────────────────
    # Phase 1 (iter 1-1000):   vs random opponents — learn basic hand values
    # Phase 2 (iter 1001-3000): self-play — refine strategy against itself
    # Phase 3 (iter 3001+):    mixed training — 70% K-Best pool, 30% current
    PHASE1_END = 1000
    PHASE2_END = 3000

    if current_iter <= PHASE1_END:
        # Phase 1: random opponents, high exploration
        epsilon = 0.10
        opp_nets = None  # random actions
    elif current_iter <= PHASE2_END:
        # Phase 2: self-play, moderate exploration
        epsilon = max(0.02, 0.05 * (1.0 - (current_iter - PHASE1_END) / (PHASE2_END - PHASE1_END)))
        opp_nets = _SHARED_STRAT_NETS
    else:
        # Phase 3: mixed training against diverse pool, low exploration
        epsilon = 0.01
        opp_nets = _SHARED_STRAT_NETS
        if _KBEST_POOL and len(_KBEST_POOL) > 0 and np.random.random() < 0.7:
            _, opp_nets = _KBEST_POOL[np.random.randint(len(_KBEST_POOL))]

    # Archetype opponent selection (per-traversal)
    archetype_skew_fn = None
    archetype_profile = None
    if _ARCHETYPE_MODE:
        arch_name = random_archetype()
        archetype_skew_fn = get_archetype_skew(arch_name)
        archetype_profile = get_archetype_profile(arch_name)

    def _pick_opponent_action(ai, obs, legal, eng=None, is_preflop=False):
        # Preflop override for opponents too
        if is_preflop and _PREFLOP_RANGES is not None and eng is not None:
            cards, rel_pos, pot_bb = _get_preflop_info(eng, ai, num_players)
            if cards and len(cards) >= 2:
                strat = _PREFLOP_RANGES.get_strategy(
                    cards, rel_pos, pot_bb, legal, _ACT_SIZE)
                lp = strat[legal].copy()
                s = lp.sum()
                lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
                return int(np.random.choice(legal, p=lp))

        nets = opp_nets
        if nets is not None:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                strat = nets[ai](obs_t).numpy()[0]
            # Apply archetype skew to opponent strategy
            if archetype_skew_fn is not None:
                strat = archetype_skew_fn(strat, legal)
                lp = strat[legal].copy()
            else:
                lp = np.maximum(strat[legal], 0.0)
            s = lp.sum()
            lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
            return int(np.random.choice(legal, p=lp))
        return int(np.random.choice(legal))

    def _run_to_terminal(eng):
        """Play to terminal using opponent strategy network. Returns reward."""
        while not eng.is_done():
            ca = eng.current_agent
            ai = eng.agents.index(ca)
            obs = eng.get_observation()
            legal = eng.get_legal_actions()
            if not legal:
                break
            is_pf = (eng.get_stage() == 'Preflop') if _PREFLOP_RANGES else False
            eng.step(_pick_opponent_action(ai, obs, legal, eng, is_pf))
        rw = eng.get_rewards()
        reward = float(rw[eng.agents[player_idx]])
        _return_engine(eng)
        return reward

    def _fresh_engine_at(history):
        """Reuse pooled engine with saved state, replay history, return positioned engine."""
        eng = _get_engine(num_players)
        if archetype_profile is not None:
            eng.set_opponent_profile(archetype_profile)
        g   = eng._get_game()
        g.np_random.set_state(pre_reset_game_rng)
        g.dealer_id = pre_reset_dealer_id
        eng.reset()
        for a in history:
            eng.step(a)
        return eng

    def _branch_exact(history, legal_int, n_samples=3):
        """Branch all legal actions, average n_samples rollouts to reduce variance."""
        results = {}
        for ba in legal_int:
            n = n_samples
            results[ba] = float(np.mean([
                _run_to_terminal(_fresh_engine_at(history + [ba]))
                for _ in range(n)
            ]))
        return results

    def _strategy_for(obs, legal_int):
        """Regret-matching strategy from advantage network."""
        if _SHARED_ADV_NETS is not None:
            with torch.no_grad():
                obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                adv_np = _SHARED_ADV_NETS[player_idx](obs_t).numpy()[0]
            return _regret_match(adv_np, legal_int, epsilon=epsilon)
        strat = np.zeros(_ACT_SIZE, dtype=np.float32)
        strat[legal_int] = 1.0 / len(legal_int)
        return strat

    def _make_steps(obs, legal_int, branch_rewards, strategy):
        """Turn branch rewards + strategy into buffer-ready step dicts.

        Regret normalization: divide by max absolute regret so values are
        in [-1, 1] range. This prevents outlier all-in regrets (±1.0 after
        _REWARD_SCALE) from dominating the advantage network's loss.
        Combined with Huber loss, this makes training robust to variance.
        """
        ev = sum(strategy[a] * branch_rewards[a] / _REWARD_SCALE for a in legal_int)
        legal_mask = np.zeros(_ACT_SIZE, dtype=np.float32)
        legal_mask[legal_int] = 1.0

        # Compute raw regrets
        raw_regrets = {a: float(np.clip(branch_rewards[a] / _REWARD_SCALE - ev,
                                         -_REGRET_CLIP, _REGRET_CLIP))
                       for a in legal_int}

        # Normalize: divide by max absolute regret (keeps direction, scales to [-1,1])
        max_abs = max(abs(r) for r in raw_regrets.values())
        if max_abs > 1e-6:
            norm_regrets = {a: r / max_abs for a, r in raw_regrets.items()}
        else:
            norm_regrets = raw_regrets  # all near zero, don't divide

        return [{
            'player':   player_idx,
            'obs':      obs,
            'action':   a,
            'legal':    legal_int,
            'regret':   float(norm_regrets[a]),
            'strategy': strategy,
            'legal_mask': legal_mask,
        } for a in legal_int]

    # ── Save game state for deterministic replay ────────────────────────────
    # With the fast (pokers) engine, game.np_random.get_state() returns the
    # immutable State object.  We must save it AFTER reset() so branches
    # replay the same dealt cards.  (Before reset returns the previous game.)
    engine = _get_engine(num_players)
    if archetype_profile is not None:
        engine.set_opponent_profile(archetype_profile)
    engine.reset()
    game = engine._get_game()
    pre_reset_game_rng  = game.np_random.get_state()
    pre_reset_dealer_id = game.dealer_id

    # ── Play opponent actions until traversal player's first decision ────────
    history = []   # grows throughout traversal; used for exact replay per branch

    while not engine.is_done():
        ca    = engine.current_agent
        ai    = engine.agents.index(ca)
        obs   = engine.get_observation()
        legal = engine.get_legal_actions()
        if not legal:
            break
        if ai == player_idx:
            break
        is_pf = (engine.get_stage() == 'Preflop') if _PREFLOP_RANGES else False
        action = _pick_opponent_action(ai, obs, legal, engine, is_pf)
        history.append(action)
        engine.step(action)

    # Check if traversal player ever got to act
    if engine.is_done():
        _return_engine(engine)
        return []
    ca    = engine.current_agent
    ai    = engine.agents.index(ca)
    if ai != player_idx:
        _return_engine(engine)
        return []

    first_obs   = engine.get_observation().copy()
    first_legal = [int(a) for a in engine.get_legal_actions()]
    _return_engine(engine)

    # ── Exact external sampling at every traversal player decision ───────────
    # For each decision: branch all actions → exact rewards → exact regrets.
    # Then sample one action from strategy and advance.
    all_steps = []

    # Preflop decision — use range override if available
    branch_rewards = _branch_exact(history, first_legal)

    if _PREFLOP_RANGES is not None:
        # Use range chart for preflop strategy (both for action selection
        # and as the strategy stored in the buffer for strategy network training)
        # Need a fresh engine positioned at current state to read cards
        probe_eng = _fresh_engine_at(history)
        cards, rel_pos, pot_bb = _get_preflop_info(probe_eng, player_idx, num_players)
        _return_engine(probe_eng)

        if cards and len(cards) >= 2:
            strategy = _PREFLOP_RANGES.get_strategy(
                cards, rel_pos, pot_bb, first_legal, _ACT_SIZE)
        else:
            strategy = _strategy_for(first_obs, first_legal)
    else:
        strategy = _strategy_for(first_obs, first_legal)

    all_steps += _make_steps(first_obs, first_legal, branch_rewards, strategy)

    lp = strategy[first_legal].copy(); lp /= lp.sum()
    history.append(int(np.random.choice(first_legal, p=lp)))

    # Remaining decisions (postflop, or additional preflop actions like re-raises)
    eng = _fresh_engine_at(history)
    while not eng.is_done():
        ca    = eng.current_agent
        ai    = eng.agents.index(ca)
        obs   = eng.get_observation()
        legal = eng.get_legal_actions()
        if not legal:
            break

        is_pf = (eng.get_stage() == 'Preflop') if _PREFLOP_RANGES else False

        if ai == player_idx:
            legal_int      = [int(a) for a in legal]
            branch_rewards = _branch_exact(history, legal_int)

            if is_pf:
                # Still preflop (e.g. facing 3bet after our open) — use range override
                cards, rel_pos, pot_bb = _get_preflop_info(eng, player_idx, num_players)
                if cards and len(cards) >= 2:
                    strategy = _PREFLOP_RANGES.get_strategy(
                        cards, rel_pos, pot_bb, legal_int, _ACT_SIZE)
                else:
                    strategy = _strategy_for(obs.copy(), legal_int)
            else:
                strategy = _strategy_for(obs.copy(), legal_int)

            all_steps += _make_steps(obs.copy(), legal_int, branch_rewards, strategy)

            lp = strategy[legal_int].copy(); lp /= lp.sum()
            action = int(np.random.choice(legal_int, p=lp))
        else:
            action = _pick_opponent_action(ai, obs, legal, eng, is_pf)

        history.append(action)
        eng.step(action)

    _return_engine(eng)
    return all_steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Deep CFR — multi-core CPU traversal + GPU training')
    parser.add_argument('--resume',  type=str,   default=None, help='Checkpoint to resume from')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Supervised model (.pt) to warm-start strategy networks from')
    parser.add_argument('--workers', type=int,   default=None, help='CPU worker count (default: cpu_count - 1)')
    parser.add_argument('--device',  type=str,   default=None, help='cuda | cpu (auto-detected if omitted)')
    parser.add_argument('--lr',      type=float, default=None, help='Override learning rate after resume')
    parser.add_argument('--preflop-ranges', action='store_true', default=True,
                        help='Lock preflop play to GTO ranges from configs/preflop_ranges.json (default: on)')
    parser.add_argument('--no-preflop-ranges', action='store_true',
                        help='Disable preflop range override, let neural net learn preflop')
    parser.add_argument('--archetype-mode', action='store_true',
                        help='Mixed archetype opponent training (extends obs to 55 dims)')
    parser.add_argument('--fine-tune-from', type=str, default=None,
                        help='Base checkpoint to fine-tune from (extends obs 50→55, lower LR)')
    parser.add_argument('--kbest-seed', type=str, nargs='+', default=None,
                        help='Checkpoint(s) to seed K-Best opponent pool (e.g. --kbest-seed models/best.pt models/iter200.pt)')
    parser.add_argument('--load-buffer', type=str, default=None,
                        help='Load replay buffer from .npz file (without resuming model weights)')
    parser.add_argument('--rake', type=float, default=0.0,
                        help='Rake percentage (e.g. 0.05 = 5%%, default: 0 = no rake)')
    parser.add_argument('--rake-cap', type=float, default=6.0,
                        help='Max rake in chips per hand (default: 6 = 3BB)')
    parser.add_argument('--equity-model', type=str, default=None,
                        help='Path to GPU equity model (.pt) to replace Monte Carlo')
    args = parser.parse_args()

    if args.no_preflop_ranges:
        args.preflop_ranges = False
    # --fine-tune-from implies --archetype-mode
    if args.fine_tune_from:
        args.archetype_mode = True

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))

    device      = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    num_players = config['game']['num_players']

    # Set globals so _make_engine() uses them in workers
    global _ARCHETYPE_MODE, _RAKE_PCT, _RAKE_CAP
    _ARCHETYPE_MODE = args.archetype_mode
    _RAKE_PCT = args.rake
    _RAKE_CAP = args.rake_cap

    # Load GPU equity model if specified
    if args.equity_model:
        from src.engine.fast_enhanced_engine import load_equity_model
        eq_device = 'cpu'  # workers run on CPU; use 'cuda' only if single-process
        load_equity_model(args.equity_model, device=eq_device)
        print(f'  equity  : GPU model ({Path(args.equity_model).name}) on {eq_device}')

    print('=' * 60)
    print('DEEP CFR  —  GPU + MULTI-CORE')
    print('=' * 60)
    print(f'  device   : {device}')
    if device == 'cuda':
        print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  engine   : fast (Rust)')
    print(f'  preflop  : {"GTO ranges (locked)" if args.preflop_ranges else "neural net (learned)"}')
    if args.rake > 0:
        print(f'  rake     : {args.rake*100:.1f}% (cap {args.rake_cap:.0f} chips = {args.rake_cap/2:.1f}BB)')
    if args.archetype_mode:
        print(f'  archetype: ON (mixed opponent training)')
    if args.fine_tune_from:
        print(f'  fine-tune: {Path(args.fine_tune_from).name}')
    if args.warm_start:
        print(f'  warm-start: {Path(args.warm_start).name}')
    print(f'  workers  : {num_workers}')
    print(f'  players  : {num_players}')
    print(f'  iters    : {config["training"]["num_iterations"]}')
    print(f'  phases   : 1-1000 random, 1001-3000 self-play, 3001+ mixed (K-Best)')
    print(f'  trav/iter: {config["training"]["traversals_per_iteration"]}')
    print('=' * 60)

    engine = _make_engine(num_players)
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()
    print(f'\n  obs_size={obs_size}  act_size={act_size}\n')

    # For fine-tuning: create DeepCFR with OLD obs size so checkpoint loads,
    # then extend to new size after loading.
    init_obs_size = obs_size
    if args.fine_tune_from:
        init_obs_size = obs_size - OPPONENT_PROFILE_DIM  # 55 - 5 = 50
        print(f'  Fine-tune mode: init obs={init_obs_size}, target obs={obs_size}')

    deep_cfr = DeepCFR(
        observation_size=init_obs_size,
        action_size=act_size,
        num_players=num_players,
        hidden_layers=config['network']['hidden_layers'],
        learning_rate=config['training']['learning_rate'],
        buffer_size=config['training']['buffer_size'],
        gradient_clip=config['training'].get('gradient_clip', 1.0),
        lr_scheduler_t_max=config['training'].get('lr_scheduler_t_max', 2000),
        dropout=config['network'].get('dropout', 0.1),
        device=device,
    )

    start_iter = 0
    cfg_lr = config['training']['learning_rate']
    cfg_t_max = config['training'].get('lr_scheduler_t_max', 2000)

    if args.fine_tune_from:
        ft_path = Path(args.fine_tune_from)
        if not ft_path.exists():
            print(f'Fine-tune checkpoint not found: {ft_path}')
            sys.exit(1)
        deep_cfr.load_models(str(ft_path))
        start_iter = deep_cfr.iteration
        print(f'Loaded base checkpoint — iteration {start_iter}')

        # Extend networks from 50 → 55 dims
        deep_cfr.extend_first_layer(obs_size)
        print(f'Extended first layer: {init_obs_size} → {obs_size} dims')

        # Lower LR for fine-tuning (5x lower than config)
        ft_lr = args.lr if args.lr else cfg_lr * 0.2
        for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
            for pg in opt.param_groups:
                pg['lr'] = ft_lr

        from torch.optim import lr_scheduler
        deep_cfr.advantage_schedulers = [
            lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=cfg_t_max)
            for opt in deep_cfr.advantage_optimizers
        ]
        deep_cfr.strategy_schedulers = [
            lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=cfg_t_max)
            for opt in deep_cfr.strategy_optimizers
        ]
        print(f'Fine-tune LR={ft_lr}, schedulers rebuilt (T_max={cfg_t_max})')

    elif args.resume:
        deep_cfr.load_models(args.resume)
        start_iter = deep_cfr.iteration
        print(f'Resumed from checkpoint — iteration {start_iter}')

        # Load replay buffer if available
        buf_path = args.resume.replace('.pt', '_buffer.npz')
        if Path(buf_path).exists():
            deep_cfr.replay_buffer.load(buf_path)
            print(f'Loaded replay buffer: {len(deep_cfr.replay_buffer):,} entries from {buf_path}')
        else:
            print(f'No buffer file found ({buf_path}), starting with empty buffer')

        # Override LR: use --lr flag if given, otherwise use config value
        new_lr = args.lr if args.lr else cfg_lr
        for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
            for pg in opt.param_groups:
                pg['lr'] = new_lr

        # Rebuild schedulers from config so new T_max / LR take effect
        from torch.optim import lr_scheduler
        deep_cfr.advantage_schedulers = [
            lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=cfg_t_max)
            for opt in deep_cfr.advantage_optimizers
        ]
        deep_cfr.strategy_schedulers = [
            lr_scheduler.LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=cfg_t_max)
            for opt in deep_cfr.strategy_optimizers
        ]
        print(f'LR set to {new_lr}, schedulers rebuilt (T_max={cfg_t_max})')

    if args.warm_start and not args.resume:
        ws_path = Path(args.warm_start)
        if not ws_path.exists():
            print(f'Warm-start file not found: {ws_path}')
            sys.exit(1)
        ckpt = torch.load(str(ws_path), map_location=device)
        if 'model_state_dict' not in ckpt:
            print(f'Warning: no model_state_dict in {ws_path.name}, skipping warm-start')
        else:
            state = ckpt['model_state_dict']
            for i, net in enumerate(deep_cfr.strategy_networks):
                net.load_state_dict(state)
            print(f'Warm-start: loaded supervised model into all {num_players} strategy networks')
            print(f'  source: {ws_path.name}  epoch={ckpt.get("epoch", "?")}  '
                  f'val_acc={ckpt.get("val_acc", 0):.1%}')

    # Load buffer from file (without resuming model weights)
    if args.load_buffer:
        buf_path = Path(args.load_buffer)
        if buf_path.exists():
            deep_cfr.replay_buffer.load(str(buf_path))
            print(f'Loaded replay buffer: {len(deep_cfr.replay_buffer):,} entries from {buf_path}')
        else:
            print(f'Warning: buffer file not found: {buf_path}')

    # Shared-memory CPU copies so workers can read current network weights.
    shared_adv_nets = []
    for net in deep_cfr.advantage_networks:
        cpu_net = copy.deepcopy(net).cpu().eval()
        cpu_net.share_memory()
        shared_adv_nets.append(cpu_net)

    shared_strat_nets = []
    for net in deep_cfr.strategy_networks:
        cpu_net = copy.deepcopy(net).cpu().eval()
        cpu_net.share_memory()
        shared_strat_nets.append(cpu_net)

    num_iterations = config['training']['num_iterations']
    traversals    = config['training']['traversals_per_iteration']
    batch_size    = config['training'].get('batch_size', 256)
    train_passes  = config['training'].get('train_passes', 20)
    ckpt_interval = config['training']['checkpoint_interval']
    models_dir    = Path(config['paths']['models'])
    models_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(config['paths'].get('logs', 'data/logs')) / 'tensorboard'
    logs_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(logs_dir))
    print(f'TensorBoard logs → {logs_dir}  (run: tensorboard --logdir {logs_dir})\n')

    # ── Set Phase 1 LR schedule: 0.0003 → 0.0001 over 1000 iters ──────
    # Override the constructor's default scheduler (which uses config t_max)
    if start_iter < 1000:
        from torch.optim import lr_scheduler as _lr_sched
        p1_lr, p1_end, p1_dur = 0.0003, 0.0001, 1000
        for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
            for pg in opt.param_groups:
                pg['lr'] = p1_lr
        deep_cfr.advantage_schedulers = [
            _lr_sched.LinearLR(opt, start_factor=1.0,
                               end_factor=p1_end / p1_lr,
                               total_iters=p1_dur)
            for opt in deep_cfr.advantage_optimizers
        ]
        deep_cfr.strategy_schedulers = [
            _lr_sched.LinearLR(opt, start_factor=1.0,
                               end_factor=p1_end / p1_lr,
                               total_iters=p1_dur)
            for opt in deep_cfr.strategy_optimizers
        ]
        # Step scheduler to match current iteration (for resume mid-Phase 1)
        for _ in range(start_iter):
            deep_cfr.step_schedulers()
        print(f'  Phase 1 LR: {p1_lr} → {p1_end} over {p1_dur} iters')

    print(f'Training iterations {start_iter + 1} → {num_iterations}')
    print(f'batch_size={batch_size}  train_passes={train_passes}')
    print(f'Spawning {num_workers} workers...')

    # K-Best Self-Play: pool of historical opponent networks
    KBEST_K = 5  # keep top-K historical versions
    kbest_pool = []  # list of (label, [shared_net_per_player])

    # Seed the pool with the best known model + resume checkpoints
    def _load_pool_checkpoint(path, label):
        """Load a checkpoint into K-Best pool as shared-memory nets."""
        from src.training.deep_cfr import StrategyNetwork
        cp_data = torch.load(str(path), map_location='cpu')
        strat_states = cp_data.get('strategy_networks', [])
        if not strat_states:
            return None
        pool_nets = []
        for sd in strat_states:
            net = StrategyNetwork(obs_size, act_size,
                                   config['network']['hidden_layers'],
                                   config['network'].get('dropout', 0.1))
            net.load_state_dict(sd)
            net.eval()
            net.share_memory()
            pool_nets.append(net)
        return (label, pool_nets)

    # Seed pool from --kbest-seed flag, or auto-detect best model
    if args.kbest_seed:
        for seed_path in args.kbest_seed:
            p = Path(seed_path)
            if p.exists():
                try:
                    entry = _load_pool_checkpoint(p, p.stem)
                    if entry:
                        kbest_pool.append(entry)
                except Exception as e:
                    print(f'  Warning: failed to load pool seed {p}: {e}')
            else:
                print(f'  Warning: pool seed not found: {p}')
    else:
        # Auto-seed with best known model if it exists
        best_model = models_dir.parent / 'deep_cfr_best.pt'
        if best_model.exists():
            try:
                entry = _load_pool_checkpoint(best_model, 'best_iter300')
                if entry:
                    kbest_pool.append(entry)
            except Exception:
                pass

    if args.resume:
        ckpt_dir = models_dir
        existing = sorted(ckpt_dir.glob('checkpoint_iter_*.pt'),
                          key=lambda p: int(p.stem.split('_')[-1]))
        if existing:
            step = max(1, len(existing) // KBEST_K)
            for cp in existing[::step][:KBEST_K]:
                try:
                    entry = _load_pool_checkpoint(cp, cp.stem)
                    if entry:
                        kbest_pool.append(entry)
                except Exception:
                    pass
            if kbest_pool:
                print(f'K-Best pool seeded with {len(kbest_pool)} checkpoints: '
                      f'{", ".join(l for l, _ in kbest_pool)}')

    pool = Pool(processes=num_workers, initializer=_worker_init,
                initargs=(shared_adv_nets, shared_strat_nets, act_size,
                          args.preflop_ranges, args.archetype_mode,
                          kbest_pool if kbest_pool else None))
    print('Workers ready.\n')

    _adv_hist   = deque(maxlen=10)
    _strat_hist = deque(maxlen=10)

    # ── Batched logging ──────────────────────────────────────────────────
    LOG_INTERVAL = 20       # print avg loss every N iterations
    EVAL_INTERVAL = 200     # self-play + validation every N iterations
    EVAL_HANDS = 2000       # hands per self-play eval (~6s)
    _batch_adv = []
    _batch_strat = []
    _batch_trav = []
    _batch_train = []

    def _trend(val, hist):
        if len(hist) < 3:
            return ' '
        avg = sum(hist) / len(hist)
        if avg == 0:
            return ' '
        r = val / avg
        if r > 1.20: return '↑!'
        if r > 1.05: return '↑ '
        if r < 0.95: return '↓ '
        return '→ '

    def _run_eval(deep_cfr_ref, eval_engine, iter_num_snap):
        """Run self-play + validation inline. Wrapped in try/except — never crashes training."""
        try:
            import io
            from src.decision_engine import DecisionEngine, DecisionConfig
            from scripts.analysis.self_play import run_self_play, generate_json

            dc = DecisionConfig(
                strong_hand_guard=False, allin_filter=False,
                style_tuning=False, use_preflop_ranges=True,
                raise_randomize=False,
            )
            sp_engine = DecisionEngine(dc)
            tmp_ckpt = str(models_dir / '_eval_tmp.pt')
            deep_cfr_ref.save_models(tmp_ckpt)
            sp_engine.load_models(facing_bet=tmp_ckpt, checked_to=tmp_ckpt)
            sp_engine.load_preflop_ranges()

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                raw_stats = run_self_play(sp_engine, EVAL_HANDS, num_players)
            finally:
                sys.stdout = old_stdout
            sp = generate_json(raw_stats, num_players, EVAL_HANDS)

            act = sp.get('postflop_actions', {})
            print(
                f"  [self-play {EVAL_HANDS}h] "
                f"VPIP={sp['vpip']:.1f}% PFR={sp['pfr']:.1f}% "
                f"CBet={sp['cbet']:.1f}% AF={sp['af']:.1f} "
                f"WTSD={sp['wtsd']:.1f}% W$SD={sp['wsd']:.1f}% "
                f"| Fold={act.get('fold_pct',0):.0f}% "
                f"Chk={act.get('check_pct',0):.0f}% "
                f"Call={act.get('call_pct',0):.0f}% "
                f"Bet={act.get('bet_raise_pct',0):.0f}%"
            )

            # TensorBoard
            for k in ('vpip', 'pfr', 'cbet', 'af', 'wtsd', 'wsd'):
                writer.add_scalar(f'selfplay/{k}', sp[k], iter_num_snap)

            try:
                os.remove(tmp_ckpt)
            except OSError:
                pass
        except Exception as e:
            print(f'  [self-play ERROR: {e}]')

        try:
            val_results = run_validation(deep_cfr_ref, eval_engine,
                                          num_games=500, verbose=False)
            print(f'  {validation_summary_line(val_results)}')

            for spot_key, spot_data in val_results.items():
                if spot_key == '_summary' or not isinstance(spot_data, dict):
                    continue
                if spot_data.get('n', 0) == 0:
                    continue
                writer.add_scalar(f'val/{spot_key}/fold', spot_data['fold_pct'], iter_num_snap)
                writer.add_scalar(f'val/{spot_key}/call', spot_data['call_pct'], iter_num_snap)
                writer.add_scalar(f'val/{spot_key}/aggr', spot_data['aggr_pct'], iter_num_snap)
            summary = val_results.get('_summary', {})
            if summary.get('total', 0) > 0:
                writer.add_scalar('val/pass_rate',
                                  summary['passed'] / summary['total'], iter_num_snap)
        except Exception as e:
            print(f'  [validation ERROR: {e}]')

    def _push_weights():
        """Copy GPU-trained weights into shared-memory CPU nets for workers."""
        for gpu_net, cpu_net in zip(deep_cfr.advantage_networks, shared_adv_nets):
            cpu_net.load_state_dict({k: v.cpu() for k, v in gpu_net.state_dict().items()})
        for gpu_net, cpu_net in zip(deep_cfr.strategy_networks, shared_strat_nets):
            cpu_net.load_state_dict({k: v.cpu() for k, v in gpu_net.state_dict().items()})

    def _build_work(it):
        return [(player_idx, num_players, it)
                for player_idx in range(num_players)
                for _ in range(traversals)]

    try:
        async_result = None   # pending traversal results (pipeline overlap)

        _last_config_reload = 0

        _phase_transitions_done = set()

        for iteration in range(start_iter, num_iterations):
            iter_num = iteration + 1
            t0 = time.time()

            # ── Phase transitions: clear buffer + reset LR ──────────────────
            #
            # Per-phase LR schedule (start → end, linear decay):
            #   Phase 1 (1-1000):    0.0003 → 0.0001
            #   Phase 2 (1001-3000): 0.0003 → 0.0001
            #   Phase 3 (3001-5000): 0.0002 → 0.0001
            #   Phase 3 (3001+):     0.0002 → 0.0001
            PHASE_LR = {
                1001: (0.0003, 0.0001, 2000),   # (start_lr, end_lr, duration)
                3001: (0.0002, 0.0001, 2000),
            }
            phase_transitions = {1001: 'P2:self-play', 3001: 'P3:mixed'}
            for phase_start, phase_name in phase_transitions.items():
                if iter_num == phase_start and phase_start not in _phase_transitions_done:
                    _phase_transitions_done.add(phase_start)
                    old_size = len(deep_cfr.replay_buffer)
                    deep_cfr.replay_buffer._size = 0
                    deep_cfr.replay_buffer._total_seen = 0
                    deep_cfr.replay_buffer._dirty = True
                    print(f'  *** Buffer cleared for {phase_name} '
                          f'(was {old_size:,} entries) ***')
                    if phase_start == 3001:
                        kbest_pool.clear()
                        print(f'  *** K-Best pool cleared for Phase 3 ***')

                    # Reset LR + rebuild scheduler for this phase
                    if phase_start in PHASE_LR:
                        from torch.optim import lr_scheduler
                        p_lr, p_end, p_dur = PHASE_LR[phase_start]
                        end_factor = p_end / p_lr
                        for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
                            for pg in opt.param_groups:
                                pg['lr'] = p_lr
                        deep_cfr.advantage_schedulers = [
                            lr_scheduler.LinearLR(opt, start_factor=1.0,
                                                  end_factor=end_factor,
                                                  total_iters=p_dur)
                            for opt in deep_cfr.advantage_optimizers
                        ]
                        deep_cfr.strategy_schedulers = [
                            lr_scheduler.LinearLR(opt, start_factor=1.0,
                                                  end_factor=end_factor,
                                                  total_iters=p_dur)
                            for opt in deep_cfr.strategy_optimizers
                        ]
                        print(f'  *** LR reset: {p_lr} → {p_end} over {p_dur} iters ***')

            # ── Hot-reload config every 10 iterations ───────────────────────
            if iter_num - _last_config_reload >= 10:
                _last_config_reload = iter_num
                try:
                    new_config = load_config(str(config_path))
                    new_tp = new_config['training'].get('train_passes', train_passes)
                    changed = []
                    if new_tp != train_passes:
                        train_passes = new_tp
                        changed.append(f'train_passes={train_passes}')
                    new_bs = new_config['training'].get('batch_size', batch_size)
                    if new_bs != batch_size:
                        batch_size = new_bs
                        changed.append(f'batch_size={batch_size}')
                    if changed:
                        print(f'  [config reload] {", ".join(changed)}')
                except Exception:
                    pass  # config file might be mid-write

            # ── Collect traversal results ────────────────────────────────────
            if async_result is not None:
                # Pipeline: traversals were launched during previous iter's training
                raw = async_result.get()
                async_result = None
            else:
                # First iteration: push weights and traverse synchronously
                _push_weights()
                print(f'Iter {iter_num:4d}  traversing...', end='\r', flush=True)
                raw = pool.map(_traversal_worker, _build_work(iter_num), chunksize=16)
            t_trav = time.time() - t0

            # ── Add all precomputed regrets to the replay buffer ─────────────
            for steps in raw:
                for step in steps:
                    deep_cfr.replay_buffer.add(
                        step['obs'], step['action'], step['regret'],
                        step['strategy'], step['player'], iteration=iter_num,
                        legal_mask=step.get('legal_mask')
                    )

            # ── Pipeline: push weights & launch next traversals BEFORE training
            # Workers use current trained weights. Training below updates GPU nets
            # but not shared memory, so no race condition. Next iter's traversals
            # are one training step behind — negligible for Deep CFR convergence.
            if iteration + 1 < num_iterations:
                _push_weights()
                async_result = pool.map_async(
                    _traversal_worker, _build_work(iter_num + 1), chunksize=16)

            # ── Train networks (GPU) — overlaps with next traversals (CPU) ───
            # Buffer warmup: after resume the buffer starts empty. Training on a
            # tiny buffer causes catastrophic overfitting (network sees all data
            # many times per iteration). Skip training entirely until buffer has
            # enough diverse samples, then ramp up train_passes gradually.
            # Minimum threshold: 10× batch_size (~655K entries, ~15 iterations).
            buf_len = len(deep_cfr.replay_buffer)
            min_buf = batch_size * 40  # ~2.6M entries, several iterations of diverse data
            if buf_len < min_buf:
                # Skip training — just collect data
                effective_passes = 0
                warmup_tag = f'  FILLING BUFFER {buf_len}/{min_buf} ({100*buf_len//min_buf}%)'
            else:
                effective_passes = train_passes
                warmup_tag = ''

            if effective_passes > 0:
                print(f'Iter {iter_num:4d}  training...        ', end='\r', flush=True)
            deep_cfr.replay_buffer.rebuild_indices()
            adv_losses, strat_losses = [], []
            for _ in range(effective_passes):
                adv_losses.append(deep_cfr.train_advantage_network(batch_size))
                strat_losses.append(deep_cfr.train_strategy_network(batch_size))
            deep_cfr.step_schedulers()
            deep_cfr.iteration = iter_num
            t_train = time.time() - t0 - t_trav

            adv   = np.mean(adv_losses)   if adv_losses   else 0.0
            strat = np.mean(strat_losses) if strat_losses else 0.0
            total = time.time() - t0

            _adv_hist.append(adv)
            _strat_hist.append(strat)
            _batch_adv.append(adv)
            _batch_strat.append(strat)
            _batch_trav.append(t_trav)
            _batch_train.append(t_train)

            if not warmup_tag:
                warmup_tag = ''
            # Phase indicator
            if iter_num <= 1000:
                phase_tag = ' P1:random'
            elif iter_num <= 3000:
                phase_tag = ' P2:self-play'
            else:
                phase_tag = ' P3:mixed'

            # ── TensorBoard: per-iteration (always) ─────────────────────────
            writer.add_scalar('loss/advantage',  adv,   iter_num)
            writer.add_scalar('loss/strategy',   strat, iter_num)
            writer.add_scalar('perf/trav_time',  t_trav,  iter_num)
            writer.add_scalar('perf/train_time', t_train, iter_num)
            writer.add_scalar('perf/total_time', total,   iter_num)
            writer.add_scalar('buffer/size', len(deep_cfr.replay_buffer), iter_num)

            # ── Print avg loss every LOG_INTERVAL iters ─────────────────────
            if iter_num % LOG_INTERVAL == 0 or warmup_tag:
                avg_a = np.mean(_batch_adv) if _batch_adv else 0.0
                avg_s = np.mean(_batch_strat) if _batch_strat else 0.0
                avg_tv = np.mean(_batch_trav) if _batch_trav else 0.0
                avg_tr = np.mean(_batch_train) if _batch_train else 0.0
                cur_lr = deep_cfr.advantage_optimizers[0].param_groups[0]['lr']
                print(
                    f'Iter {iter_num:4d}/{num_iterations}'
                    f'  buf={len(deep_cfr.replay_buffer):>7d}'
                    f'  adv={avg_a:.4f}{_trend(avg_a, _adv_hist)}'
                    f'  strat={avg_s:.4f}{_trend(avg_s, _strat_hist)}'
                    f'  lr={cur_lr:.5f}'
                    f'  trav={avg_tv:.1f}s'
                    f'  train={avg_tr:.1f}s'
                    f'{warmup_tag}{phase_tag}'
                )
                _batch_adv.clear()
                _batch_strat.clear()
                _batch_trav.clear()
                _batch_train.clear()

            # ── K-Best pool update every 100 iters, only in Phase 2+ ────────
            if iter_num % 100 == 0 and iter_num > 1000:
                try:
                    pool_nets = []
                    for net in deep_cfr.strategy_networks:
                        pnet = copy.deepcopy(net).cpu().eval()
                        pnet.share_memory()
                        pool_nets.append(pnet)
                    label = f'iter_{iter_num}'
                    kbest_pool.append((label, pool_nets))
                    if len(kbest_pool) > KBEST_K:
                        kbest_pool = [kbest_pool[0]] + kbest_pool[-(KBEST_K - 1):]
                    # Restart workers with updated pool
                    if async_result is not None:
                        async_result.get()
                        async_result = None
                    pool.terminate()
                    pool.join()
                    pool = Pool(processes=num_workers, initializer=_worker_init,
                                initargs=(shared_adv_nets, shared_strat_nets, act_size,
                                          args.preflop_ranges, args.archetype_mode,
                                          kbest_pool))
                    pool_labels = ', '.join(l for l, _ in kbest_pool)
                    print(f'  K-Best pool updated: [{pool_labels}]')
                except Exception as e:
                    print(f'  K-Best pool update failed: {e}')

            # ── Checkpoint save ─────────────────────────────────────────────
            if iter_num % ckpt_interval == 0:
                ckpt = str(models_dir / f'checkpoint_iter_{iter_num}.pt')
                deep_cfr.save_models(ckpt)
                buf_path = ckpt.replace('.pt', '_buffer.npz')
                deep_cfr.replay_buffer.save(buf_path)
                meta = {'iteration': iter_num,
                        'buffer_size': len(deep_cfr.replay_buffer),
                        'config': config['training']}
                with open(ckpt.replace('.pt', '_metadata.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

            # ── Self-play + validation every EVAL_INTERVAL iters ────────────
            if iter_num % EVAL_INTERVAL == 0:
                _run_eval(deep_cfr, engine, iter_num)
    finally:
        if async_result is not None:
            async_result.wait()
        pool.terminate()
        pool.join()
        writer.close()

    final = str(models_dir / 'final_model_enhanced.pt')
    deep_cfr.save_models(final)
    print(f'\nTraining complete. Model saved to {final}')
    engine.close()


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    mp.set_start_method('spawn', force=True)
    main()
