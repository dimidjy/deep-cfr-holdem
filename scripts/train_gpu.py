"""
Deep CFR training: parallel CPU traversals + batched GPU training.

Workers run full external-sampling MCCFR traversals covering all streets:
  - Preflop: exact external sampling (branch all actions, exact terminal rewards)
  - Postflop: outcome sampling with network bootstrapping for unsampled actions
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

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config
from validate_spots import run_validation, validation_summary_line


# Shared-memory CPU networks — set once per worker, updated by main process each iteration.
_SHARED_ADV_NETS   = None
_SHARED_STRAT_NETS = None
_ACT_SIZE          = None   # action space size, needed for uniform fallback strategy

_REWARD_SCALE = 100.0
_REGRET_CLIP  = 2.0


def _worker_init(shared_adv, shared_strat, act_size):
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE
    _SHARED_ADV_NETS  = shared_adv
    _SHARED_STRAT_NETS = shared_strat
    _ACT_SIZE          = act_size
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    os.environ['GYMNASIUM_DISABLE_LOGGING'] = '1'


def _regret_match(adv, legal, epsilon=0.05):
    """Regret-matching with epsilon-uniform exploration."""
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    strategy[legal] = pos / s if s > 0 else 1.0 / len(legal)
    strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
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
    )


def _traversal_worker(args):
    """
    External-sampling MCCFR traversal covering all streets.

    Phase 1-3 (preflop): expand ALL legal actions at the traversal player's
    first preflop decision, get exact terminal rewards per branch → exact regrets.

    Phase 4 (postflop): sample one preflop action from the current strategy,
    continue the game, and collect outcome-sampling regrets for every subsequent
    traversal player decision (flop/turn/river).
      - Sampled action regret:   terminal_reward / REWARD_SCALE - ev_network
      - Unsampled action regret: adv_network[a]  - ev_network  (bootstrapped)

    Both game.np_random and game.dealer_id are saved/restored for each branch
    to guarantee identical card deals and action ordering across all branches.
    """
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE
    player_idx, num_players = args

    def _pick_opponent_action(ai, obs, legal):
        nets = _SHARED_STRAT_NETS
        if nets is not None:
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                strat = nets[ai](obs_t).numpy()[0]
            lp = np.maximum(strat[legal], 0.0)
            s = lp.sum()
            lp = lp / s if s > 0 else np.ones(len(legal)) / len(legal)
            return int(np.random.choice(legal, p=lp))
        return int(np.random.choice(legal))

    def _run_to_terminal(eng):
        """Play to terminal with outcome sampling for all players. Returns reward."""
        while not eng.is_done():
            ca = eng.current_agent
            ai = eng.agents.index(ca)
            obs = eng.get_observation()
            legal = eng.get_legal_actions()
            if not legal:
                break
            eng.step(_pick_opponent_action(ai, obs, legal))
        rw = eng.get_rewards()
        reward = float(rw[eng.agents[player_idx]])
        eng.close()
        return reward

    def _run_collecting_postflop(eng):
        """Play to terminal, collecting traversal player's decisions for regret computation.

        Traversal player uses regret-matching on current advantage network.
        Opponents use strategy network. Returns (decisions, terminal_reward).
        """
        decisions = []
        while not eng.is_done():
            ca    = eng.current_agent
            ai    = eng.agents.index(ca)
            obs   = eng.get_observation()
            legal = eng.get_legal_actions()
            if not legal:
                break
            if ai == player_idx:
                if _SHARED_ADV_NETS is not None:
                    with torch.no_grad():
                        obs_t  = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                        adv_np = _SHARED_ADV_NETS[player_idx](obs_t).numpy()[0]
                else:
                    adv_np = np.zeros(_ACT_SIZE, dtype=np.float32)
                legal_int = [int(a) for a in legal]
                strategy  = _regret_match(adv_np, legal_int)
                lp = strategy[legal_int]
                lp = lp / lp.sum()
                action = int(np.random.choice(legal_int, p=lp))
                decisions.append({
                    'obs':      obs.copy(),
                    'action':   action,
                    'legal':    legal_int,
                    'adv':      adv_np.copy(),
                    'strategy': strategy.copy(),
                })
            else:
                action = _pick_opponent_action(ai, obs, legal)
            eng.step(action)
        rw = eng.get_rewards()
        terminal_reward = float(rw[eng.agents[player_idx]])
        eng.close()
        return decisions, terminal_reward

    def _restore_engine():
        """Create a fresh engine with the same card deal and button position."""
        eng = _make_engine(num_players)
        g   = eng._get_game()
        g.np_random.set_state(pre_reset_game_rng)
        g.dealer_id = pre_reset_dealer_id
        eng.reset()
        for stored_action in pre_actions:
            eng.step(stored_action)
        return eng

    # ── Phase 1: play until traversal player's first preflop decision ───────
    # Save game RNG state AND dealer_id before reset.
    # dealer_id (button position) is set randomly during engine creation from
    # global np.random — it is NOT part of game.np_random — so must be saved
    # separately. Without this, branches get different action orders despite
    # the same card deal, causing replayed pre-actions to go to the wrong players.
    engine = _make_engine(num_players)
    game   = engine._get_game()
    pre_reset_game_rng  = game.np_random.get_state()
    pre_reset_dealer_id = game.dealer_id
    engine.reset()

    pre_actions = []
    first_obs   = None
    first_legal = None

    while not engine.is_done():
        ca    = engine.current_agent
        ai    = engine.agents.index(ca)
        obs   = engine.get_observation()
        legal = engine.get_legal_actions()
        if not legal:
            break
        if ai == player_idx:
            first_obs   = obs.copy()
            first_legal = [int(a) for a in legal]
            break
        action = _pick_opponent_action(ai, obs, legal)
        pre_actions.append(action)
        engine.step(action)

    engine.close()

    if first_obs is None:
        return []

    # ── Phase 2: branch over all preflop actions, get exact terminal rewards ─
    branch_rewards = {}
    for branch_action in first_legal:
        eng2 = _restore_engine()
        eng2.step(branch_action)
        branch_rewards[branch_action] = _run_to_terminal(eng2)

    # ── Phase 3: exact preflop regrets ──────────────────────────────────────
    if _SHARED_ADV_NETS is not None:
        with torch.no_grad():
            obs_t  = torch.from_numpy(first_obs.astype(np.float32)).unsqueeze(0)
            adv_np = _SHARED_ADV_NETS[player_idx](obs_t).numpy()[0]
        strategy = _regret_match(adv_np, first_legal)
    else:
        strategy = np.zeros(_ACT_SIZE, dtype=np.float32)
        strategy[first_legal] = 1.0 / len(first_legal)

    ev = sum(strategy[a] * branch_rewards[a] / _REWARD_SCALE for a in first_legal)

    preflop_steps = [{
        'player':   player_idx,
        'obs':      first_obs,
        'action':   a,
        'legal':    list(first_legal),
        'regret':   float(np.clip(branch_rewards[a] / _REWARD_SCALE - ev, -_REGRET_CLIP, _REGRET_CLIP)),
        'strategy': strategy,
    } for a in first_legal]

    # ── Phase 4: postflop outcome sampling ──────────────────────────────────
    # Sample one preflop action from current strategy, then collect regrets
    # at every subsequent traversal player decision (flop/turn/river).
    lp = strategy[first_legal].copy()
    lp /= lp.sum()
    sampled_action = int(np.random.choice(first_legal, p=lp))

    eng3 = _restore_engine()
    eng3.step(sampled_action)
    postflop_decisions, terminal_reward = _run_collecting_postflop(eng3)

    # Outcome-sampling regrets:
    #   sampled action  → grounded in actual terminal reward
    #   unsampled actions → bootstrapped from advantage network
    # Both centered on ev_network so relative ordering is preserved.
    postflop_steps = []
    for dec in postflop_decisions:
        adv   = dec['adv']
        legal = dec['legal']
        strat = dec['strategy']
        ev_net = float(np.dot(strat[legal], adv[legal]))
        for a in legal:
            if a == dec['action']:
                regret = terminal_reward / _REWARD_SCALE - ev_net
            else:
                regret = float(adv[a]) - ev_net
            postflop_steps.append({
                'player':   player_idx,
                'obs':      dec['obs'],
                'action':   a,
                'legal':    legal,
                'regret':   float(np.clip(regret, -_REGRET_CLIP, _REGRET_CLIP)),
                'strategy': strat,
            })

    return preflop_steps + postflop_steps


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Deep CFR — multi-core CPU traversal + GPU training')
    parser.add_argument('--resume',  type=str, default=None, help='Checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=None, help='CPU worker count (default: cpu_count - 1)')
    parser.add_argument('--device',  type=str, default=None, help='cuda | cpu (auto-detected if omitted)')
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))

    device      = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = args.workers or max(1, mp.cpu_count() - 1)
    num_players = config['game']['num_players']

    print('=' * 60)
    print('DEEP CFR  —  GPU + MULTI-CORE')
    print('=' * 60)
    print(f'  device   : {device}')
    if device == 'cuda':
        print(f'  GPU      : {torch.cuda.get_device_name(0)}')
    print(f'  workers  : {num_workers}')
    print(f'  players  : {num_players}')
    print(f'  iters    : {config["training"]["num_iterations"]}')
    print(f'  trav/iter: {config["training"]["traversals_per_iteration"]}')
    print('=' * 60)

    engine = _make_engine(num_players)
    obs_size = engine.get_observation_space_size()
    act_size = engine.get_action_space_size()
    print(f'\n  obs_size={obs_size}  act_size={act_size}\n')

    deep_cfr = DeepCFR(
        observation_size=obs_size,
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
    if args.resume:
        deep_cfr.load_models(args.resume)
        start_iter = deep_cfr.iteration
        print(f'Resumed from checkpoint — iteration {start_iter}')

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

    print(f'Training iterations {start_iter + 1} → {num_iterations}')
    print(f'batch_size={batch_size}  train_passes={train_passes}')
    print(f'Spawning {num_workers} workers...')

    pool = Pool(processes=num_workers, initializer=_worker_init,
                initargs=(shared_adv_nets, shared_strat_nets, act_size))
    print('Workers ready.\n')

    _adv_hist   = deque(maxlen=10)
    _strat_hist = deque(maxlen=10)

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

    try:
        for iteration in range(start_iter, num_iterations):
            iter_num = iteration + 1
            t0 = time.time()

            # Push latest trained weights into shared memory for workers.
            for gpu_net, cpu_net in zip(deep_cfr.advantage_networks, shared_adv_nets):
                cpu_net.load_state_dict({k: v.cpu() for k, v in gpu_net.state_dict().items()})
            for gpu_net, cpu_net in zip(deep_cfr.strategy_networks, shared_strat_nets):
                cpu_net.load_state_dict({k: v.cpu() for k, v in gpu_net.state_dict().items()})

            # Build work items: each traversal specifies only (player_idx, num_players)
            work = [(player_idx, num_players)
                    for player_idx in range(num_players)
                    for _ in range(traversals)]

            print(f'Iter {iter_num:4d}  traversing...', end='\r', flush=True)
            raw = pool.map(_traversal_worker, work, chunksize=16)
            t_trav = time.time() - t0

            # Add all precomputed regrets to the replay buffer
            print(f'Iter {iter_num:4d}  computing regrets...', end='\r', flush=True)
            for steps in raw:
                for step in steps:
                    deep_cfr.replay_buffer.add(
                        step['obs'], step['action'], step['regret'],
                        step['strategy'], step['player'], iteration=iter_num
                    )
            t_regret = time.time() - t0 - t_trav

            # Train networks
            print(f'Iter {iter_num:4d}  training...        ', end='\r', flush=True)
            adv_losses, strat_losses = [], []
            for _ in range(train_passes):
                adv_losses.append(deep_cfr.train_advantage_network(batch_size))
                strat_losses.append(deep_cfr.train_strategy_network(batch_size))
            deep_cfr.iteration = iter_num
            t_train = time.time() - t0 - t_trav - t_regret

            adv   = np.mean(adv_losses)   if adv_losses   else 0.0
            strat = np.mean(strat_losses) if strat_losses else 0.0
            total = time.time() - t0

            _adv_hist.append(adv)
            _strat_hist.append(strat)

            print(
                f'Iter {iter_num:4d}/{num_iterations}'
                f'  buf={len(deep_cfr.replay_buffer):>7d}'
                f'  adv={adv:.4f}{_trend(adv, _adv_hist)}'
                f'  strat={strat:.4f}{_trend(strat, _strat_hist)}'
                f'  trav={t_trav:.1f}s'
                f'  train={t_train:.1f}s'
                f'  total={total:.1f}s'
            )

            if iter_num % ckpt_interval == 0:
                ckpt = str(models_dir / f'checkpoint_iter_{iter_num}.pt')
                deep_cfr.save_models(ckpt)
                meta = {'iteration': iter_num,
                        'buffer_size': len(deep_cfr.replay_buffer),
                        'config': config['training']}
                with open(ckpt.replace('.pt', '_metadata.json'), 'w') as f:
                    json.dump(meta, f, indent=2)

                val_results = run_validation(deep_cfr, engine, num_games=500, verbose=False)
                print(f'  {validation_summary_line(val_results)}')
    finally:
        pool.terminate()
        pool.join()

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
