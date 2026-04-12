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


# =============================================================================
# Training phase boundaries
# =============================================================================
PHASE1_END = 600    # iters 1..600 — random opponents (learn basic hand values)
PHASE2_END = 1600   # iters 601..1600 — pure self-play
                    # iters 1601+     — mixed (K-Best pool + self-play)

# Per-phase LR schedule: (start_lr, end_lr, duration_iters)
PHASE_LR = {
    601:  (0.0003, 0.0001, 1000),
    1601: (0.0002, 0.0001, 3600),
}

# =============================================================================
# Reward/regret constants
# =============================================================================
_REWARD_SCALE = 100.0   # divides chip rewards → regrets in roughly [-1, 1]
_REGRET_CLIP  = 1.0     # clip bound on regret after scaling
_RAKE_PCT     = 0.0
_RAKE_CAP     = 0.0

# =============================================================================
# Worker-process globals — set via _worker_init, read by _traversal_worker
# =============================================================================
# Neural networks (shared memory, updated each iteration by main)
_SHARED_ADV_NETS   = None   # advantage nets, one per player
_SHARED_STRAT_NETS = None   # strategy nets, one per player
_ACT_SIZE          = None   # action space size

# Opponent pool + preflop ranges
_KBEST_POOL         = None  # [(label, [net_per_player])] — historical opponents
_PREFLOP_RANGES     = None  # hero preflop range charts
_OPP_PREFLOP_RANGES = None  # wider NL5 ranges for opponents (if --nl5-opponents)

# Training modes (flags)
_ARCHETYPE_MODE = False     # mixed archetype opponents (extends obs 50→55)
_NL5_OPP_MODE   = False     # force NL5 population skew on all opponents
_DECK_RESHUFFLE = False     # reshuffle undealt cards per branch rollout
_BLUFF_PENALTY  = 0.0       # penalize raises with equity<0.5 (value-only nudge)
_CALL_BONUS     = 0.0       # reward +EV calls (equity > pot_odds_required)

# Engine pool — per-worker reuse to avoid env teardown on every branch
_ENGINE_POOL = []


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
                 archetype_mode=False, kbest_pool=None, equity_model_path=None,
                 nl5_opp_mode=False, deck_reshuffle=False, bluff_penalty=0.0,
                 call_bonus=0.0):
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE, _PREFLOP_RANGES
    global _ARCHETYPE_MODE, _KBEST_POOL, _OPP_PREFLOP_RANGES, _NL5_OPP_MODE
    global _DECK_RESHUFFLE, _BLUFF_PENALTY, _CALL_BONUS
    _SHARED_ADV_NETS  = shared_adv
    _SHARED_STRAT_NETS = shared_strat
    _ACT_SIZE          = act_size
    _ARCHETYPE_MODE    = archetype_mode
    _KBEST_POOL        = kbest_pool
    _NL5_OPP_MODE      = nl5_opp_mode
    _DECK_RESHUFFLE    = deck_reshuffle
    _BLUFF_PENALTY     = bluff_penalty
    _CALL_BONUS        = call_bonus
    if use_preflop_ranges:
        from src.preflop_ranges import PreflopRanges
        _PREFLOP_RANGES = PreflopRanges()
        # Load wider NL5 preflop ranges for opponents
        if nl5_opp_mode:
            from pathlib import Path
            nl5_path = str(Path(__file__).parent.parent / 'configs' / 'preflop_ranges_nl5.json')
            _OPP_PREFLOP_RANGES = PreflopRanges(ranges_path=nl5_path)
        else:
            _OPP_PREFLOP_RANGES = _PREFLOP_RANGES
    else:
        _PREFLOP_RANGES = None
        _OPP_PREFLOP_RANGES = None
    if equity_model_path:
        import torch as _torch
        # Each worker must use only 1 thread for PyTorch ops.
        # Without this, 127 workers × N default threads = thousands of
        # threads fighting for CPU cores → deadlock-level contention.
        _torch.set_num_threads(1)
        from src.engine.fast_enhanced_engine import load_equity_model
        load_equity_model(equity_model_path, device='cpu')
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
    """Standard regret matching with epsilon-uniform exploration.

    σ(I,a) = max(0, R(I,a)) / Σ max(0, R(I,a'))
    Plus epsilon-greedy exploration to ensure all actions are sampled.
    No call floor — with correct opponent play (advantage network),
    call receives proper regrets naturally.
    """
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    strategy[legal] = pos / s if s > 0 else 1.0 / len(legal)
    if epsilon > 0:
        strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
    return strategy


def _make_engine(num_players):
    from src.engine.factory import make_engine
    return make_engine(
        num_players=num_players,
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


def _reshuffled_init_state(init_state, num_players, n_visible_public):
    """Build a new initial pkrs.State with reshuffled future board cards.

    Preserves hole cards (so action history stays consistent) and the first
    n_visible_public board cards (which are part of the info state at the
    decision point). Cards drawn AFTER the decision (turn/river if not yet
    visible) are reshuffled — gives independent chance samples per rollout
    and per branch, drastically reducing regret variance.

    Card-to-deck-position mapping (pokers `from_deck` semantics):
    - hole cards are dealt 2-at-a-time, starting from button+1 (SB)
    - position 2*N onwards goes to the public cards in order
    """
    import pokers as pkrs
    button = init_state.button

    # Place hole cards in their correct deal-order slots
    hole_in_deal_order = [None] * (2 * num_players)
    for pid in range(num_players):
        deal_idx = (pid - button - 1) % num_players
        hole_in_deal_order[2 * deal_idx]     = init_state.players_state[pid].hand[0]
        hole_in_deal_order[2 * deal_idx + 1] = init_state.players_state[pid].hand[1]

    # init_state.deck is the 40 cards remaining after hole cards. The first
    # 5 of these become the public cards in order; the rest are burns.
    full_52 = hole_in_deal_order + list(init_state.deck)
    n_fixed = (2 * num_players) + n_visible_public
    fixed = full_52[:n_fixed]
    suffix = list(full_52[n_fixed:])
    np.random.shuffle(suffix)
    return pkrs.State.from_deck(
        n_players=num_players,
        button=button,
        sb=1, bb=2, stake=100,
        deck=fixed + suffix,
    )


def _select_phase_opponent(current_iter, kbest_pool):
    """Return (epsilon, opp_nets) for the current training phase.

    Phase 1 (iters 1..PHASE1_END):     random opponents, high exploration
    Phase 2 (iters PHASE1_END+1..PHASE2_END):  self-play vs advantage net
    Phase 3 (PHASE2_END+1..):           mixed 70% K-Best pool / 30% self-play

    NL5 mode overrides Phase 2+: always uses K-Best pool (if seeded) with
    NL5 skew applied at pick_opponent_action time. Prevents co-adaptation
    collapse into check-check-showdown equilibrium.
    """
    if current_iter <= PHASE1_END:
        return 0.10, None  # random actions

    if _NL5_OPP_MODE:
        eps = max(0.05, 0.08 * (1.0 - (current_iter - PHASE1_END) / 3000))
        if kbest_pool:
            _, opp_nets = kbest_pool[np.random.randint(len(kbest_pool))]
        else:
            opp_nets = _SHARED_STRAT_NETS  # fallback if no pool
        return eps, opp_nets

    if current_iter <= PHASE2_END:
        # Deep CFR paper: opponents play current iteration's strategy (advantage net),
        # not the accumulated average.
        eps = max(0.06, 0.10 * (1.0 - (current_iter - PHASE1_END) / (PHASE2_END - PHASE1_END)))
        return eps, _SHARED_ADV_NETS

    # Phase 3 — mixed pool
    opp_nets = _SHARED_ADV_NETS
    if kbest_pool and np.random.random() < 0.7:
        _, opp_nets = kbest_pool[np.random.randint(len(kbest_pool))]
    return 0.05, opp_nets


def _create_worker_pool(num_workers, shared_adv, shared_strat, act_size,
                        args, kbest_pool, eq_path, nl5_opp):
    """Spawn a worker Pool wired with all current training state.

    Used both for initial pool creation and after K-Best updates (which
    terminate + rebuild the pool so workers see the new opponent list).
    """
    return Pool(
        processes=num_workers, initializer=_worker_init,
        initargs=(shared_adv, shared_strat, act_size,
                  args.preflop_ranges, args.archetype_mode,
                  kbest_pool if kbest_pool else None,
                  eq_path, nl5_opp, args.deck_reshuffle,
                  args.bluff_penalty, args.call_bonus),
    )


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
    player_idx, num_players, current_iter = args

    # Phase-dependent opponent + exploration ε (see _select_phase_opponent)
    epsilon, opp_nets = _select_phase_opponent(current_iter, _KBEST_POOL)

    # Archetype opponent selection (per-traversal)
    archetype_skew_fn = None
    archetype_profile = None
    if _ARCHETYPE_MODE:
        arch_name = random_archetype()
        archetype_skew_fn = get_archetype_skew(arch_name)
        archetype_profile = get_archetype_profile(arch_name)

    def _pick_opponent_action(ai, obs, legal, eng=None, is_preflop=False):
        # Preflop override for opponents — use NL5 ranges if in nl5 mode
        if is_preflop and _PREFLOP_RANGES is not None and eng is not None:
            opp_ranges = _OPP_PREFLOP_RANGES if _OPP_PREFLOP_RANGES else _PREFLOP_RANGES
            cards, rel_pos, pot_bb = _get_preflop_info(eng, ai, num_players)
            if cards and len(cards) >= 2:
                strat = opp_ranges.get_strategy(
                    cards, rel_pos, pot_bb, legal, _ACT_SIZE)
                lp = strat[legal].copy()
                s = lp.sum()
                lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
                return int(np.random.choice(legal, p=lp))

        nets = opp_nets
        if nets is not None:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                adv = nets[ai](obs_t).numpy()[0]
            # Apply regret matching on advantage output (same as traverser)
            # This is what Deep CFR paper specifies for opponent play:
            # σ(I,a) = max(0, R(I,a)) / Σ max(0, R(I,a'))
            strat = _regret_match(adv, legal, epsilon=epsilon)
            # Apply archetype skew if configured
            if _NL5_OPP_MODE and archetype_skew_fn is None:
                from src.archetypes import nl5_population_skew
                strat = nl5_population_skew(strat, legal)
            elif archetype_skew_fn is not None:
                strat = archetype_skew_fn(strat, legal)
            lp = strat[legal].copy()
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

    def _fresh_engine_at(history, init_state=None):
        """Reuse pooled engine with saved state, replay history, return positioned engine.

        If init_state is provided, use it instead of pre_reset_game_rng — used for
        deck reshuffle, where each branch rollout gets a state with re-randomized
        future board cards (hole cards and revealed-so-far public cards preserved).
        """
        eng = _get_engine(num_players)
        if archetype_profile is not None:
            eng.set_opponent_profile(archetype_profile)
        g   = eng._get_game()
        g.np_random.set_state(init_state if init_state is not None else pre_reset_game_rng)
        g.dealer_id = pre_reset_dealer_id
        eng.reset()
        for a in history:
            eng.step(a)
        return eng

    def _branch_exact(history, legal_int, n_samples=3, n_visible_public=0,
                      hero_equity=None, pot_chips=None, pot_odds=None):
        """Branch all legal actions, average n_samples rollouts to reduce variance.

        With _DECK_RESHUFFLE on, each rollout uses an independent reshuffle of
        undealt cards (turn/river not yet visible). This is the variance-reduction
        trick from Steinberger Deep-CFR (MultiOutcomeSampler.reshuffle_remaining_deck).

        With _BLUFF_PENALTY > 0, raise-action rewards get a penalty proportional
        to (0.5 - hero_equity) * pot. This shifts the Nash equilibrium toward
        value-only play — bluffs need to overcome the penalty to stay in strategy.

        With _CALL_BONUS > 0, the call action reward gets a bonus proportional
        to how much equity exceeds pot odds (i.e. the call is +EV in a vacuum).
        Counterbalances the bluff penalty so defender doesn't drift too nitty,
        which would otherwise inflate fold equity for the aggressor.

        pot_odds: obs[10] — defined as pot / (pot + call). 0 means no call amount
                  (free check). Required equity to call = 1 - pot_odds.
        """
        results = {}

        bluff_penalty_chips = 0.0
        if _BLUFF_PENALTY > 0 and hero_equity is not None and pot_chips is not None:
            deficit = max(0.0, 0.5 - float(hero_equity))
            bluff_penalty_chips = deficit * float(pot_chips) * _BLUFF_PENALTY

        # Call bonus: reward +EV calls (equity > required pot-odds equity).
        # post_call_pot = pot + call = pot / pot_odds (rearranged from obs formula).
        call_bonus_chips = 0.0
        if (_CALL_BONUS > 0 and hero_equity is not None and pot_chips is not None
                and pot_odds is not None and float(pot_odds) > 0):
            po = float(pot_odds)
            required_eq = 1.0 - po
            excess = max(0.0, float(hero_equity) - required_eq)
            post_call_pot = float(pot_chips) / max(po, 1e-6)
            call_bonus_chips = excess * post_call_pot * _CALL_BONUS

        for ba in legal_int:
            rewards = []
            for _ in range(n_samples):
                init = (_reshuffled_init_state(pre_reset_game_rng, num_players, n_visible_public)
                        if _DECK_RESHUFFLE else None)
                rewards.append(_run_to_terminal(_fresh_engine_at(history + [ba], init_state=init)))
            r = float(np.mean(rewards))
            if bluff_penalty_chips > 0 and ba >= 2:
                r -= bluff_penalty_chips
            if call_bonus_chips > 0 and ba == 1:
                r += call_bonus_chips
            results[ba] = r
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

        # Store raw regrets — no normalization, no call attenuation.
        # With correct opponent play (advantage network with regret matching),
        # call receives proper regrets naturally.
        # Previous code divided by max absolute regret, which destroyed
        # magnitude information — marginal spots looked identical to clear
        # spots. The Huber loss + gradient clipping handle outliers instead.
        norm_regrets = raw_regrets

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
    first_n_public = len(engine._state.public_cards)
    first_pot_chips = engine.get_pot_size()
    _return_engine(engine)

    # ── Exact external sampling at every traversal player decision ───────────
    # For each decision: branch all actions → exact rewards → exact regrets.
    # Then sample one action from strategy and advance.
    all_steps = []

    # Preflop decision — use range override if available.
    # obs[10]=pot_odds, obs[18]=hero equity (see CLAUDE.md observation layout).
    branch_rewards = _branch_exact(history, first_legal,
                                    n_visible_public=first_n_public,
                                    hero_equity=float(first_obs[18]),
                                    pot_chips=first_pot_chips,
                                    pot_odds=float(first_obs[10]))

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
            n_pub          = len(eng._state.public_cards)
            branch_rewards = _branch_exact(history, legal_int,
                                            n_visible_public=n_pub,
                                            hero_equity=float(obs[18]),
                                            pot_chips=eng.get_pot_size(),
                                            pot_odds=float(obs[10]))

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


# =============================================================================
# Main-process helpers (setup, used from main())
# =============================================================================

def _rebuild_linear_lr_schedulers(deep_cfr, end_factor, total_iters):
    """Rebuild LinearLR schedulers for both advantage + strategy optimizers."""
    from torch.optim import lr_scheduler
    deep_cfr.advantage_schedulers = [
        lr_scheduler.LinearLR(opt, start_factor=1.0,
                               end_factor=end_factor, total_iters=total_iters)
        for opt in deep_cfr.advantage_optimizers
    ]
    deep_cfr.strategy_schedulers = [
        lr_scheduler.LinearLR(opt, start_factor=1.0,
                               end_factor=end_factor, total_iters=total_iters)
        for opt in deep_cfr.strategy_optimizers
    ]


def _set_all_optimizers_lr(deep_cfr, lr):
    """Set lr on every param group of every advantage + strategy optimizer."""
    for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
        for pg in opt.param_groups:
            pg['lr'] = lr


def _apply_fine_tune(deep_cfr, args, cfg_lr, cfg_t_max, obs_size, init_obs_size):
    """Load base checkpoint, extend obs layer, drop LR to 0.2× for fine-tune.

    Returns start_iter (from the loaded checkpoint).
    """
    ft_path = Path(args.fine_tune_from)
    if not ft_path.exists():
        print(f'Fine-tune checkpoint not found: {ft_path}')
        sys.exit(1)
    deep_cfr.load_models(str(ft_path))
    start_iter = deep_cfr.iteration
    print(f'Loaded base checkpoint — iteration {start_iter}')

    deep_cfr.extend_first_layer(obs_size)
    print(f'Extended first layer: {init_obs_size} → {obs_size} dims')

    ft_lr = args.lr if args.lr else cfg_lr * 0.2
    _set_all_optimizers_lr(deep_cfr, ft_lr)
    _rebuild_linear_lr_schedulers(deep_cfr, end_factor=0.1, total_iters=cfg_t_max)
    print(f'Fine-tune LR={ft_lr}, schedulers rebuilt (T_max={cfg_t_max})')
    return start_iter


def _apply_resume(deep_cfr, args, cfg_lr, cfg_t_max):
    """Load checkpoint + buffer, set LR, rebuild schedulers. Returns start_iter."""
    deep_cfr.load_models(args.resume)
    start_iter = deep_cfr.iteration
    print(f'Resumed from checkpoint — iteration {start_iter}')

    buf_path = args.resume.replace('.pt', '_buffer.npz')
    if Path(buf_path).exists():
        deep_cfr.replay_buffer.load(buf_path)
        print(f'Loaded replay buffer: {len(deep_cfr.replay_buffer):,} entries from {buf_path}')
    else:
        print(f'No buffer file found ({buf_path}), starting with empty buffer')

    new_lr = args.lr if args.lr else cfg_lr
    _set_all_optimizers_lr(deep_cfr, new_lr)
    _rebuild_linear_lr_schedulers(deep_cfr, end_factor=0.1, total_iters=cfg_t_max)
    print(f'LR set to {new_lr}, schedulers rebuilt (T_max={cfg_t_max})')
    return start_iter


def _apply_warm_start(deep_cfr, args, num_players, device):
    """Load supervised model into all strategy networks (only if not resuming)."""
    ws_path = Path(args.warm_start)
    if not ws_path.exists():
        print(f'Warm-start file not found: {ws_path}')
        sys.exit(1)
    ckpt = torch.load(str(ws_path), map_location=device)
    if 'model_state_dict' not in ckpt:
        print(f'Warning: no model_state_dict in {ws_path.name}, skipping warm-start')
        return
    state = ckpt['model_state_dict']
    for net in deep_cfr.strategy_networks:
        net.load_state_dict(state)
    print(f'Warm-start: loaded supervised model into all {num_players} strategy networks')
    print(f'  source: {ws_path.name}  epoch={ckpt.get("epoch", "?")}  '
          f'val_acc={ckpt.get("val_acc", 0):.1%}')


def _share_networks_to_cpu(nets):
    """Deep-copy GPU nets to CPU + share_memory() for multiprocessing."""
    shared = []
    for net in nets:
        cpu_net = copy.deepcopy(net).cpu().eval()
        cpu_net.share_memory()
        shared.append(cpu_net)
    return shared


def _setup_phase1_lr(deep_cfr, start_iter):
    """Phase 1 LR schedule: 0.0003 → 0.0001 over 600 iters (only if in Phase 1)."""
    if start_iter >= 600:
        return
    p1_lr, p1_end, p1_dur = 0.0003, 0.0001, 600
    _set_all_optimizers_lr(deep_cfr, p1_lr)
    _rebuild_linear_lr_schedulers(deep_cfr, end_factor=p1_end / p1_lr, total_iters=p1_dur)
    # Step scheduler to match current iteration (for resume mid-Phase 1)
    for _ in range(start_iter):
        deep_cfr.step_schedulers()
    print(f'  Phase 1 LR: {p1_lr} → {p1_end} over {p1_dur} iters')


# =============================================================================
# Training-loop helpers (used from main())
# =============================================================================

_PHASE_NAMES = {601: 'P2:self-play', 1601: 'P3:mixed'}


def _handle_phase_transition(iter_num, deep_cfr, done_set):
    """On Phase 2/3 boundary: clear replay buffer + reset LR schedule.

    Called once per iteration; no-op unless iter_num is a phase start AND
    that phase hasn't already been transitioned. `done_set` tracks which
    phase boundaries have already fired (prevents double-reset on resume).

    Pool is NOT cleared — keeps diverse models from earlier phases.
    """
    phase_name = _PHASE_NAMES.get(iter_num)
    if phase_name is None or iter_num in done_set:
        return
    done_set.add(iter_num)

    old_size = len(deep_cfr.replay_buffer)
    deep_cfr.replay_buffer._size = 0
    deep_cfr.replay_buffer._total_seen = 0
    deep_cfr.replay_buffer._dirty = True
    print(f'  *** Buffer cleared for {phase_name} (was {old_size:,} entries) ***')

    if iter_num in PHASE_LR:
        from torch.optim import lr_scheduler
        p_lr, p_end, p_dur = PHASE_LR[iter_num]
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


# =============================================================================
# Inline evaluation helpers (used inside training loop)
# =============================================================================

def _selfplay_eval_stats(deep_cfr_ref, num_players, num_hands, eq_path):
    """Run `num_hands` of self-play and return postflop action-mix stats.

    Preflop locked to GTO ranges. Postflop uses deep_cfr.get_strategy()
    directly (same path as validation, avoids DecisionEngine obs mismatch).

    Returns dict with: af, fold_pct, chk_pct, call_pct, bet_pct.
    """
    import pokers as _pkrs_local
    from src.engine.factory import make_engine as _mk_eng
    from src.preflop_ranges import PreflopRanges

    if eq_path:
        from src.engine.fast_enhanced_engine import load_equity_model
        load_equity_model(eq_path, device='cpu')

    sp_eng = _mk_eng(num_players=num_players)
    sp_ranges = PreflopRanges()
    postflop = {'bets': 0, 'calls': 0, 'checks': 0, 'folds': 0}

    for _ in range(num_hands):
        sp_eng.reset()
        while not sp_eng.is_done():
            ca = sp_eng.current_agent
            ai = sp_eng.agents.index(ca)
            obs = sp_eng.get_observation()
            legal = sp_eng.get_legal_actions()
            if not legal:
                break
            legal_int = [int(a) for a in legal]
            stage = sp_eng.get_stage()

            # Preflop = GTO ranges; postflop = model
            if stage == 'Preflop':
                cards = sp_eng.get_player_cards(ca)
                if cards and len(cards) >= 2:
                    rel = (ai - sp_eng._button) % num_players
                    pot = sp_eng.get_pot_size() / 2.0
                    strat = sp_ranges.get_strategy(cards, rel, pot, legal_int, 6)
                else:
                    strat = np.zeros(6, dtype=np.float32)
                    strat[legal_int] = 1.0 / len(legal_int)
            else:
                raw = deep_cfr_ref.get_strategy(obs, ai % num_players, legal_int)
                strat = np.zeros(6, dtype=np.float32)
                for a in legal_int:
                    if a < len(raw):
                        strat[a] = max(0.0, float(raw[a]))
                s = strat.sum()
                if s > 0:
                    strat /= s
                else:
                    strat[legal_int] = 1.0 / len(legal_int)

            # Sample action
            probs = np.array([strat[a] for a in legal_int], dtype=np.float64)
            probs = np.maximum(probs, 0.0)
            ps = probs.sum()
            probs = probs / ps if ps > 0 else np.ones(len(legal_int)) / len(legal_int)
            action = int(np.random.choice(legal_int, p=probs))

            # Track postflop action mix
            if stage != 'Preflop':
                if action == 0:
                    postflop['folds'] += 1
                elif action == 1:
                    if _pkrs_local.ActionEnum.Check in sp_eng._state.legal_actions:
                        postflop['checks'] += 1
                    else:
                        postflop['calls'] += 1
                else:
                    postflop['bets'] += 1

            sp_eng.step(action)

    t_acts = max(sum(postflop.values()), 1)
    return {
        'af':       postflop['bets'] / max(postflop['calls'], 1),
        'fold_pct': 100 * postflop['folds']  / t_acts,
        'chk_pct':  100 * postflop['checks'] / t_acts,
        'call_pct': 100 * postflop['calls']  / t_acts,
        'bet_pct':  100 * postflop['bets']   / t_acts,
    }


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
    parser.add_argument('--deck-reshuffle', action='store_true',
                        help='Reshuffle undealt cards per branch rollout (variance reduction). '
                             'Each branch gets independent turn/river — Steinberger MultiOutcomeSampler trick.')
    parser.add_argument('--bluff-penalty', type=float, default=0.0,
                        help='Penalty coefficient for raising with equity < 0.5. '
                             '0 = off, 0.3 = soft (allows semi-bluffs), 0.6 = medium, '
                             '1.0 = value-only play. Applied to raise-action rewards as '
                             '(0.5 - equity) * pot * coefficient.')
    parser.add_argument('--call-bonus', type=float, default=0.0,
                        help='Bonus coefficient for +EV calls (equity > required pot-odds '
                             'equity). 0 = off, 0.3 = soft, 0.6 = medium. Applied to call '
                             'action reward as max(0, equity - (1-pot_odds)) * (pot+call) * '
                             'coefficient. Symmetric to --bluff-penalty — use together so the '
                             'defender doesn\'t drift too nitty when penalty cuts raising.')
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
    parser.add_argument('--nl5-opponents', action='store_true',
                        help='Opponents use NL5 population behavior: wider preflop ranges + '
                             'passive postflop skew (less folding, more calling, less raising)')
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
    if args.nl5_opponents:
        print(f'  nl5-opp  : ON (NL5 preflop ranges + passive postflop skew)')
    if args.fine_tune_from:
        print(f'  fine-tune: {Path(args.fine_tune_from).name}')
    if args.warm_start:
        print(f'  warm-start: {Path(args.warm_start).name}')
    print(f'  workers  : {num_workers}')
    print(f'  players  : {num_players}')
    print(f'  iters    : {config["training"]["num_iterations"]}')
    print(f'  phases   : 1-600 random, 601-1600 self-play, 1601+ mixed (K-Best)')
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

    cfg_lr = config['training']['learning_rate']
    cfg_t_max = config['training'].get('lr_scheduler_t_max', 2000)

    # Checkpoint loading (at most one of fine-tune/resume applies)
    if args.fine_tune_from:
        start_iter = _apply_fine_tune(deep_cfr, args, cfg_lr, cfg_t_max,
                                       obs_size, init_obs_size)
    elif args.resume:
        start_iter = _apply_resume(deep_cfr, args, cfg_lr, cfg_t_max)
    else:
        start_iter = 0

    # Warm-start supervised weights (only when not resuming from CFR checkpoint)
    if args.warm_start and not args.resume:
        _apply_warm_start(deep_cfr, args, num_players, device)

    # Load replay buffer from file (without resuming model weights)
    if args.load_buffer:
        buf_path = Path(args.load_buffer)
        if buf_path.exists():
            deep_cfr.replay_buffer.load(str(buf_path))
            print(f'Loaded replay buffer: {len(deep_cfr.replay_buffer):,} entries from {buf_path}')
        else:
            print(f'Warning: buffer file not found: {buf_path}')

    # Shared-memory CPU copies so workers can read current network weights
    shared_adv_nets   = _share_networks_to_cpu(deep_cfr.advantage_networks)
    shared_strat_nets = _share_networks_to_cpu(deep_cfr.strategy_networks)

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

    # Phase 1 LR schedule (0.0003 → 0.0001 over 600 iters) — overrides the
    # DeepCFR constructor's default scheduler. Noop if past Phase 1.
    _setup_phase1_lr(deep_cfr, start_iter)

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

    # Auto-discover K-Best pool from model dir on resume, BUT only when the
    # user didn't specify --kbest-seed explicitly (otherwise we'd double up).
    if args.resume and not args.kbest_seed:
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

    _eq_path = args.equity_model if args.equity_model else None
    _nl5_opp = getattr(args, 'nl5_opponents', False)
    pool = _create_worker_pool(num_workers, shared_adv_nets, shared_strat_nets,
                                act_size, args, kbest_pool, _eq_path, _nl5_opp)
    print(f'Workers ready. deck_reshuffle={args.deck_reshuffle} '
          f'bluff_penalty={args.bluff_penalty} call_bonus={args.call_bonus}\n')

    _adv_hist   = deque(maxlen=10)
    _strat_hist = deque(maxlen=10)

    # ── Batched logging ──────────────────────────────────────────────────
    LOG_INTERVAL = 20       # print avg loss every N iterations
    EVAL_INTERVAL = 200     # self-play + validation every N iterations
    EVAL_HANDS = 2000       # hands per self-play eval
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
        """Self-play + validation eval, guarded so training never crashes."""
        # ── Self-play postflop mix ──────────────────────────────────────
        try:
            stats = _selfplay_eval_stats(deep_cfr_ref, num_players, EVAL_HANDS, _eq_path)
            print(
                f"  [self-play {EVAL_HANDS}h] AF={stats['af']:.2f} "
                f"| Fold={stats['fold_pct']:.0f}% Chk={stats['chk_pct']:.0f}% "
                f"Call={stats['call_pct']:.0f}% Bet={stats['bet_pct']:.0f}%"
            )
            for k in ('af', 'fold_pct', 'chk_pct', 'call_pct', 'bet_pct'):
                writer.add_scalar(f'selfplay/{k}', stats[k], iter_num_snap)
        except Exception as e:
            print(f'  [self-play ERROR: {e}]')

        # ── Spot validation (AA vs 72o fold rates etc.) ─────────────────
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

            # Phase transitions: buffer clear + LR reset (Phase 2 & 3 boundaries)
            _handle_phase_transition(iter_num, deep_cfr, _phase_transitions_done)

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
                # Reinitialize advantage networks from scratch every 10 iterations.
                # Paper-correct version is per-iteration; relaxing to every 10 keeps
                # the bias-prevention effect while letting each network actually
                # converge on the buffer between resets.
                if iter_num % 10 == 0:
                    deep_cfr.reinitialize_advantage_networks()
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
            if iter_num <= 600:
                phase_tag = ' P1:random'
            elif iter_num <= 1600:
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

            # ── K-Best pool update (skip in NL5 mode — no pool used) ─────
            if iter_num % 100 == 0 and iter_num > 1600 and not _nl5_opp:
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
                    pool = _create_worker_pool(num_workers, shared_adv_nets,
                                               shared_strat_nets, act_size,
                                               args, kbest_pool, _eq_path, _nl5_opp)
                    pool_labels = ', '.join(l for l, _ in kbest_pool)
                    print(f'  K-Best pool updated: [{pool_labels}]')
                except Exception as e:
                    print(f'  K-Best pool update failed: {e}')

            # ── Checkpoint save ─────────────────────────────────────────────
            if iter_num % ckpt_interval == 0:
                ckpt = str(models_dir / f'checkpoint_iter_{iter_num}.pt')
                deep_cfr.save_models(ckpt)
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
