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

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config
from validate_spots import run_validation, validation_summary_line


# Shared-memory CPU networks — set once per worker, updated by main process each iteration.
_SHARED_ADV_NETS   = None
_SHARED_STRAT_NETS = None
_ACT_SIZE          = None   # action space size, needed for uniform fallback strategy

_REWARD_SCALE = 100.0
_REGRET_CLIP  = 1.0

# Worker-local engine pool — reuse PettingZoo envs instead of recreating per branch
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
    if epsilon > 0:
        strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
    return strategy


_USE_FAST_ENGINE = False   # set from main() based on --engine flag


def _make_engine(num_players):
    if _USE_FAST_ENGINE:
        from src.engine import FastEnhancedPokerEngine
        return FastEnhancedPokerEngine(
            num_players=num_players,
            add_position_features=True,
            add_pot_odds_features=True,
            add_hand_strength_features=True,
            add_stack_features=True,
            add_equity_features=True,
            add_street_features=True,
            add_game_state_features=True,
            add_betting_history_features=True,
        )
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
    """
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _ACT_SIZE
    player_idx, num_players, current_iter = args
    # Decay exploration: 5% early → 1% by iter 2000
    epsilon = max(0.01, 0.05 * (1.0 - current_iter / 3000.0))

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
        """Play to terminal using opponent strategy network. Returns reward."""
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
        _return_engine(eng)
        return reward

    def _fresh_engine_at(history):
        """Reuse pooled engine with saved state, replay history, return positioned engine."""
        eng = _get_engine(num_players)
        g   = eng._get_game()
        g.np_random.set_state(pre_reset_game_rng)
        g.dealer_id = pre_reset_dealer_id
        eng.reset()
        for a in history:
            eng.step(a)
        return eng

    def _branch_exact(history, legal_int, n_samples=5):
        """Branch all legal actions, average n_samples rollouts to reduce variance."""
        return {
            ba: float(np.mean([
                _run_to_terminal(_fresh_engine_at(history + [ba]))
                for _ in range(n_samples)
            ]))
            for ba in legal_int
        }

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
        """Turn branch rewards + strategy into buffer-ready step dicts."""
        ev = sum(strategy[a] * branch_rewards[a] / _REWARD_SCALE for a in legal_int)
        return [{
            'player':   player_idx,
            'obs':      obs,
            'action':   a,
            'legal':    legal_int,
            'regret':   float(np.clip(branch_rewards[a] / _REWARD_SCALE - ev,
                                      -_REGRET_CLIP, _REGRET_CLIP)),
            'strategy': strategy,
        } for a in legal_int]

    # ── Save game state for deterministic replay ────────────────────────────
    # dealer_id (button position) is set from global np.random during engine
    # creation and is NOT part of game.np_random — save both separately.
    engine = _get_engine(num_players)
    game   = engine._get_game()
    pre_reset_game_rng  = game.np_random.get_state()
    pre_reset_dealer_id = game.dealer_id
    engine.reset()

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
        action = _pick_opponent_action(ai, obs, legal)
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

    # Preflop decision
    branch_rewards = _branch_exact(history, first_legal)
    strategy       = _strategy_for(first_obs, first_legal)
    all_steps     += _make_steps(first_obs, first_legal, branch_rewards, strategy)

    lp = strategy[first_legal].copy(); lp /= lp.sum()
    history.append(int(np.random.choice(first_legal, p=lp)))

    # Postflop decisions
    eng = _fresh_engine_at(history)
    while not eng.is_done():
        ca    = eng.current_agent
        ai    = eng.agents.index(ca)
        obs   = eng.get_observation()
        legal = eng.get_legal_actions()
        if not legal:
            break

        if ai == player_idx:
            legal_int      = [int(a) for a in legal]
            branch_rewards = _branch_exact(history, legal_int)
            strategy       = _strategy_for(obs.copy(), legal_int)
            all_steps     += _make_steps(obs.copy(), legal_int, branch_rewards, strategy)

            lp = strategy[legal_int].copy(); lp /= lp.sum()
            action = int(np.random.choice(legal_int, p=lp))
        else:
            action = _pick_opponent_action(ai, obs, legal)

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
    parser.add_argument('--workers', type=int,   default=None, help='CPU worker count (default: cpu_count - 1)')
    parser.add_argument('--device',  type=str,   default=None, help='cuda | cpu (auto-detected if omitted)')
    parser.add_argument('--lr',      type=float, default=None, help='Override learning rate after resume')
    parser.add_argument('--engine',  type=str,   default='pettingzoo',
                        choices=['pettingzoo', 'fast'],
                        help='Game engine backend: pettingzoo (default) or fast (Rust-backed, ~30x faster)')
    args = parser.parse_args()

    global _USE_FAST_ENGINE
    _USE_FAST_ENGINE = (args.engine == 'fast')

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
    print(f'  engine   : {"fast (Rust)" if _USE_FAST_ENGINE else "pettingzoo (RLCard)"}')
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
        if args.lr:
            for opt in deep_cfr.advantage_optimizers + deep_cfr.strategy_optimizers:
                for pg in opt.param_groups:
                    pg['lr'] = args.lr
            print(f'LR overridden to {args.lr}')

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

        for iteration in range(start_iter, num_iterations):
            iter_num = iteration + 1
            t0 = time.time()

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
                        step['strategy'], step['player'], iteration=iter_num
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
            print(f'Iter {iter_num:4d}  training...        ', end='\r', flush=True)
            deep_cfr.replay_buffer.rebuild_indices()
            adv_losses, strat_losses = [], []
            for _ in range(train_passes):
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

            print(
                f'Iter {iter_num:4d}/{num_iterations}'
                f'  buf={len(deep_cfr.replay_buffer):>7d}'
                f'  adv={adv:.4f}{_trend(adv, _adv_hist)}'
                f'  strat={strat:.4f}{_trend(strat, _strat_hist)}'
                f'  trav={t_trav:.1f}s'
                f'  train={t_train:.1f}s'
                f'  total={total:.1f}s'
            )

            # ── TensorBoard: per-iteration metrics ──────────────────────────
            writer.add_scalar('loss/advantage',  adv,   iter_num)
            writer.add_scalar('loss/strategy',   strat, iter_num)
            writer.add_scalar('perf/trav_time',  t_trav,  iter_num)
            writer.add_scalar('perf/train_time', t_train, iter_num)
            writer.add_scalar('perf/total_time', total,   iter_num)
            writer.add_scalar('buffer/size', len(deep_cfr.replay_buffer), iter_num)

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

                # ── TensorBoard: validation metrics ─────────────────────────
                for spot_key, spot_data in val_results.items():
                    if spot_key == '_summary' or not isinstance(spot_data, dict):
                        continue
                    if spot_data.get('n', 0) == 0:
                        continue
                    writer.add_scalar(f'val/{spot_key}/fold', spot_data['fold_pct'], iter_num)
                    writer.add_scalar(f'val/{spot_key}/call', spot_data['call_pct'], iter_num)
                    writer.add_scalar(f'val/{spot_key}/aggr', spot_data['aggr_pct'], iter_num)
                summary = val_results.get('_summary', {})
                if summary.get('total', 0) > 0:
                    writer.add_scalar('val/pass_rate',
                                      summary['passed'] / summary['total'], iter_num)
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
