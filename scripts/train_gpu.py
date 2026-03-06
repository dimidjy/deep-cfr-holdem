"""
High-performance Deep CFR training.

Architecture:
  - N CPU workers run game traversals in parallel (multiprocessing Pool)
  - After each iteration, main process batches ALL collected observations
    into a single GPU forward pass to compute advantages/regrets
  - GPU then runs training passes on the replay buffer

This beats cfr_trainer_parallel.py  (per-step CPU inference)
and cfr_trainer_async.py  (Manager queues, broken regret, deadlock-prone).
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

import copy
from collections import deque

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))  # so validate_spots is importable

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.utils import load_config
from validate_spots import run_validation, validation_summary_line


# Shared-memory CPU networks — set once per worker via initializer,
# then updated in-place by the main process before each iteration.
_SHARED_ADV_NETS   = None
_SHARED_STRAT_NETS = None
# Opponent pool — list of frozen strategy-network lists (one per checkpoint).
# Each game randomly picks one model from the pool for all opponent decisions.
# Best response to a diverse pool converges toward Nash equilibrium much faster
# than best response to a single fixed opponent.
_FROZEN_STRAT_POOL = None   # list[ list[nn.Module] ]  (pool × players)


def _worker_init(shared_adv, shared_strat, frozen_pool):
    """Store shared model references and suppress noisy warnings."""
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _FROZEN_STRAT_POOL
    _SHARED_ADV_NETS    = shared_adv
    _SHARED_STRAT_NETS  = shared_strat
    _FROZEN_STRAT_POOL  = frozen_pool   # None  or  list of 6-net lists
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    os.environ['GYMNASIUM_DISABLE_LOGGING'] = '1'


def _regret_match_np(adv, legal, epsilon=0.05):
    """Regret-matching strategy from advantage values, with epsilon-uniform mix."""
    strategy = np.zeros(len(adv), dtype=np.float32)
    pos = np.maximum(adv[legal], 0.0)
    s = pos.sum()
    if s > 0:
        strategy[legal] = pos / s
    else:
        strategy[legal] = 1.0 / len(legal)
    # Mix with uniform to ensure all actions (including fold) get sampled
    strategy[legal] = (1.0 - epsilon) * strategy[legal] + epsilon / len(legal)
    return strategy


# ---------------------------------------------------------------------------
# Worker (runs in subprocess — no GPU, no network weights)
# ---------------------------------------------------------------------------

def _traversal_worker(args):
    """
    Game simulator. The traversed player samples actions from the current
    regret-matching strategy (advantage nets). Opponents sample from the
    current strategy network — proper self-play MCCFR, not random.
    All regret computation still happens in the main (GPU) process.
    """
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS, _FROZEN_STRAT_POOL
    player_idx, seed, num_players = args
    np.random.seed(seed)

    from src.engine import EnhancedPokerEngine  # local import for subprocess
    engine = EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
    )
    engine.reset()

    # Pick one opponent model for the whole game (consistent within each hand).
    # Randomly sampling from the pool each game means the traversed player must
    # be good against ALL pool members, pushing strategy toward Nash equilibrium.
    if _FROZEN_STRAT_POOL is not None:
        opp_nets = _FROZEN_STRAT_POOL[np.random.randint(len(_FROZEN_STRAT_POOL))]
    else:
        opp_nets = _SHARED_STRAT_NETS

    episode_data = []
    while not engine.is_done():
        current_agent = engine.current_agent
        agent_idx = engine.agents.index(current_agent)
        obs = engine.get_observation()
        legal = engine.get_legal_actions()

        if not legal:
            break

        if agent_idx == player_idx and _SHARED_ADV_NETS is not None:
            # Traversed player: sample from current regret-matching strategy
            with torch.no_grad():
                obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                adv_np = _SHARED_ADV_NETS[agent_idx](obs_t).numpy()[0]
            strategy = _regret_match_np(adv_np, legal)
            legal_probs = strategy[legal]
            legal_probs = legal_probs / legal_probs.sum()
            action = int(np.random.choice(legal, p=legal_probs))
        else:
            # Opponent: use the game's selected pool model (or self-play nets).
            if opp_nets is not None:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                    strat = opp_nets[agent_idx](obs_t).numpy()[0]
                legal_probs = np.maximum(strat[legal], 0.0)
                s = legal_probs.sum()
                legal_probs = legal_probs / s if s > 0 else np.ones(len(legal)) / len(legal)
                action = int(np.random.choice(legal, p=legal_probs))
            else:
                action = int(np.random.choice(legal))

        episode_data.append({
            'player': agent_idx,
            'obs': obs.copy(),
            'action': action,
            'legal': [int(a) for a in legal],
        })
        engine.step(action)

    rewards = engine.get_rewards()
    final_reward = float(rewards[engine.agents[player_idx]])
    engine.close()

    return episode_data, final_reward, player_idx


# ---------------------------------------------------------------------------
# Batched GPU regret computation
# ---------------------------------------------------------------------------

# Normalise terminal rewards so the advantage network always sees values in
# roughly the same ±1 range regardless of num_players or stack size.
# PettingZoo texas_holdem_no_limit_v6 starts each player with 100 chips.
_REWARD_SCALE = 100.0


def _preflop_fold_synthetic_reward(obs, num_players):
    """
    When a non-blind player folds preflop (reward == 0), the zero reward
    creates a self-reinforcing feedback loop: network estimates play as
    negative EV → fold gets positive regret → strategy converges to fold
    even for premium hands (AA ends up folding more than 72o).

    Fix: replace the 0 reward with a small negative synthetic reward
    proportional to how much equity the player gave up by folding.
    This gives the gradient the correct direction (fold AA = bad,
    fold 72o = less bad) without changing the CFR math for any case
    where chips were actually at stake.

    Hand features are at obs[num_players+5 : num_players+10]:
      [is_pair, is_suited, high_card_norm, low_card_norm, rank_gap_norm]
    This holds for both obs_size=25 (6p no equity) and obs_size=29 (6p with equity).
    """
    hand_start = num_players + 5  # pos_oh(N) + rel_pos(1) + is_btn(1) + pot_odds(3)
    is_pair    = float(obs[hand_start])
    is_suited  = float(obs[hand_start + 1])
    high_card  = float(obs[hand_start + 2])
    low_card   = float(obs[hand_start + 3])

    # Same preflop equity formula as EnhancedPokerEngine._get_equity_features
    if is_pair > 0.5:
        equity = 0.53 + high_card * 0.32          # 22→0.53, AA→0.85
    else:
        gap     = max(0.0, high_card - low_card)
        equity  = (0.36
                   + high_card * 0.15
                   + low_card  * 0.08
                   + is_suited * 0.03
                   - gap       * 0.12)
    equity = float(np.clip(equity, 0.25, 0.85))

    # Symmetric synthetic reward: penalise folding premium hands, reward folding trash.
    # Threshold 0.5 is the equity break-even point.
    # AA  (equity=0.85): synth = (0.5-0.85)*4 = -1.4 → clipped to -1.0  (don't fold)
    # J9o (equity=0.50): synth ≈ 0                                        (neutral)
    # 72o (equity=0.37): synth = (0.5-0.37)*4 = +0.51                     (fold more)
    # 32o (equity=0.36): synth = (0.5-0.36)*4 = +0.55                     (fold more)
    # The positive reward for trash gives regret(fold)=synth-EV>0 even with a noisy
    # network, breaking the frozen-uniform-strategy problem.
    THRESHOLD = 0.50
    synthetic = (THRESHOLD - equity) * 4.0
    return float(np.clip(synthetic, -1.0, 1.0))


def _compute_regrets_gpu(episodes, deep_cfr, iteration):
    """
    Collect every (obs, legal_actions) from the traversed player's steps across
    all episodes, run ONE batched GPU forward pass per player, then compute and
    store regrets for every legal action.
    """
    # Index: entry → (ep_idx, step_idx)
    entries = []   # (ep_idx, step_idx, player_idx, obs, action, legal)
    by_player = {p: [] for p in range(deep_cfr.num_players)}

    for ep_idx, (ep_data, final_reward, trav_player) in enumerate(episodes):
        for step_idx, step in enumerate(ep_data):
            if step['player'] != trav_player:
                continue
            entry_idx = len(entries)
            entries.append((ep_idx, step_idx, step['player'],
                            step['obs'], step['action'], step['legal']))
            by_player[step['player']].append(entry_idx)

    if not entries:
        return

    # Pre-compute effective rewards.  For non-blind preflop folds (raw reward == 0)
    # replace the zero with a small equity-based negative signal to break the
    # self-reinforcing fold feedback loop in multi-player outcome sampling.
    effective_rewards = []
    for ep_data, raw_reward, trav_player in episodes:
        eff = raw_reward / _REWARD_SCALE
        if abs(raw_reward) < 0.5:   # reward ≈ 0 → possible non-blind preflop fold
            trav_steps = [s for s in ep_data if s['player'] == trav_player]
            if len(trav_steps) == 1 and trav_steps[0]['action'] == 0:
                eff = _preflop_fold_synthetic_reward(
                    trav_steps[0]['obs'], deep_cfr.num_players
                )
        effective_rewards.append(eff)

    all_advantages = [None] * len(entries)

    # One GPU forward pass per player
    for player_idx, idxs in by_player.items():
        if not idxs:
            continue
        obs_np = np.array([entries[i][3] for i in idxs], dtype=np.float32)
        obs_t = torch.from_numpy(obs_np).to(deep_cfr.device)
        with torch.no_grad():
            adv_t = deep_cfr.advantage_networks[player_idx](obs_t).cpu().numpy()
        for i, adv in zip(idxs, adv_t):
            all_advantages[i] = adv

    # Compute per-action regrets and push to replay buffer
    for entry_idx, (ep_idx, _, player_idx, obs, action, legal) in enumerate(entries):
        final_reward = effective_rewards[ep_idx]   # already /-scaled or synthetic
        adv = all_advantages[entry_idx]

        mask = np.zeros(deep_cfr.action_size, dtype=np.float32)
        mask[legal] = 1.0
        adv = adv * mask

        # Step 1: initial strategy for EV estimation (regret-matching on raw adv).
        # Strategy is ratio-based so it is scale-invariant.
        pos = np.maximum(adv, 0.0) * mask
        pos_sum = pos.sum()
        strategy_init = pos / pos_sum if pos_sum > 0 else mask / mask.sum()

        # Step 2: Normalize adv to [-1, 1] so that ev is on the same scale as
        # final_reward (which is already divided by _REWARD_SCALE).
        # Without this, ev grows with the network's output magnitude (seen as
        # adv loss explosion) while final_reward stays in [-1,+1], making
        # regret(sampled) = final_reward - ev consistently large-negative.
        # Normalization only scales down (adv_scale >= 1), never amplifies noise.
        legal_adv = adv[legal]
        adv_scale = max(float(np.abs(legal_adv).max()), 1.0)
        adv_normed = adv / adv_scale

        ev = float(np.sum(strategy_init[legal] * adv_normed[legal]))

        # Step 3: relative advantages — centers on EV so "less bad" actions
        # get positive regret even when all absolute values are negative.
        # Use adv_normed so rel_adv is in the same [-1,1] scale as final_reward.
        rel_adv = adv_normed - ev
        rel_adv = rel_adv * mask

        # Step 4: proper strategy from relative advantages (stored in buffer
        # and used to train the strategy network). Scale-invariant.
        pos2 = np.maximum(rel_adv, 0.0) * mask
        pos2_sum = pos2.sum()
        strategy = pos2 / pos2_sum if pos2_sum > 0 else mask / mask.sum()

        # Step 5: regrets — both now in the same [-1,+1] scale.
        # Sampled action: grounded in actual terminal reward.
        # Unsampled actions: bootstrapped from normalized relative advantages.
        _REGRET_CLIP = 2.0
        for a in legal:
            regret = (final_reward - ev) if a == action else float(rel_adv[a])
            regret = float(np.clip(regret, -_REGRET_CLIP, _REGRET_CLIP))
            deep_cfr.replay_buffer.add(obs, a, regret, strategy,
                                       player_idx, iteration=iteration)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Deep CFR — multi-core CPU traversal + GPU training'
    )
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from')
    parser.add_argument('--workers', type=int, default=None,
                        help='CPU worker count (default: cpu_count - 1)')
    parser.add_argument('--device', type=str, default=None,
                        help='cuda | cpu  (auto-detected if omitted)')
    parser.add_argument('--frozen-opponents', type=str, nargs='+', default=None,
                        help='One or more checkpoints used as frozen opponents '
                             '(e.g. models/6player_transferred.pt models/checkpoint_iter_100.pt). '
                             'Each game randomly picks one model from the pool, '
                             'preventing bad self-play equilibria in 6-player training.')
    args = parser.parse_args()

    config_path = Path(__file__).parent.parent / 'configs/training_config.json'
    config = load_config(str(config_path))

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
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

    engine = EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
        add_equity_features=True,
        add_street_features=True,
        add_game_state_features=True,
    )
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

    # Shared-memory CPU copies of advantage + strategy networks.
    # Workers read from these; main process writes updated weights each iteration.
    # share_memory() maps tensors to OS shared memory — zero-copy across processes.
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

    # Load frozen opponent pool if requested.
    # Each entry in the pool is a list of num_players strategy networks from one checkpoint.
    # Workers randomly pick one pool member per game → best response to a diverse
    # mixture of strategies converges toward Nash equilibrium (fictitious play).
    frozen_strat_pool = None
    if args.frozen_opponents:
        frozen_strat_pool = []
        for ckpt_path in args.frozen_opponents:
            print(f'Loading frozen opponent checkpoint: {ckpt_path}')
            frozen_ckpt = torch.load(ckpt_path, map_location='cpu')
            src_nets = frozen_ckpt['strategy_networks']
            pool_nets = []
            for i in range(num_players):
                net = copy.deepcopy(deep_cfr.strategy_networks[i]).cpu()
                net.load_state_dict(src_nets[i % len(src_nets)])
                net.eval()
                net.share_memory()
                pool_nets.append(net)
            frozen_strat_pool.append(pool_nets)
        print(f'  Frozen pool: {len(frozen_strat_pool)} checkpoint(s), {num_players} networks each.')

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
    # Create Pool ONCE — avoid re-spawning 30 processes every iteration.
    # With 'spawn' + torch, each process takes ~3s to start, so per-iter
    # Pool creation would add 3s × num_workers of overhead each iteration.
    pool = Pool(processes=num_workers, initializer=_worker_init,
                initargs=(shared_adv_nets, shared_strat_nets, frozen_strat_pool))
    print('Workers ready.\n')

    # Rolling history for trend indicators (last 10 iters)
    _adv_hist   = deque(maxlen=10)
    _strat_hist = deque(maxlen=10)
    _last_trash_fold  = None   # updated at each checkpoint validation
    _last_72o_fold    = None

    def _trend(val, hist):
        """Return arrow showing whether val is trending up/down vs recent avg."""
        if len(hist) < 3:
            return ' '
        avg = sum(hist) / len(hist)
        if avg == 0:
            return ' '
        ratio = val / avg
        if ratio > 1.20:
            return '↑!'
        if ratio > 1.05:
            return '↑ '
        if ratio < 0.95:
            return '↓ '
        return '→ '

    try:
        for iteration in range(start_iter, num_iterations):
            iter_num = iteration + 1
            t0 = time.time()

            # Push latest trained weights into shared memory so workers use them.
            # load_state_dict does an in-place copy → shared memory is updated
            # before pool.map dispatches any tasks (no concurrent read/write).
            for gpu_net, cpu_net in zip(deep_cfr.advantage_networks, shared_adv_nets):
                cpu_net.load_state_dict(
                    {k: v.cpu() for k, v in gpu_net.state_dict().items()}
                )
            for gpu_net, cpu_net in zip(deep_cfr.strategy_networks, shared_strat_nets):
                cpu_net.load_state_dict(
                    {k: v.cpu() for k, v in gpu_net.state_dict().items()}
                )

            # --- Build work items ---
            work = []
            for player_idx in range(num_players):
                for i in range(traversals):
                    seed = iter_num * 100_000 + player_idx * 10_000 + i
                    work.append((player_idx, seed, num_players))

            # --- CPU: parallel game traversals ---
            print(f'Iter {iter_num:4d}  traversing...', end='\r', flush=True)
            episodes = pool.map(_traversal_worker, work, chunksize=16)
            t_trav = time.time() - t0

            # --- GPU: batch regret computation ---
            print(f'Iter {iter_num:4d}  computing regrets...', end='\r', flush=True)
            _compute_regrets_gpu(episodes, deep_cfr, iter_num)
            t_regret = time.time() - t0 - t_trav

            # --- GPU: network training ---
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

            adv_arrow   = _trend(adv,   _adv_hist)
            strat_arrow = _trend(strat, _strat_hist)
            _adv_hist.append(adv)
            _strat_hist.append(strat)

            fold_str = ''
            if _last_trash_fold is not None:
                fold_str = f'  trash={_last_trash_fold:.0%}'
            if _last_72o_fold is not None:
                fold_str += f'  72o={_last_72o_fold:.0%}'

            print(
                f'Iter {iter_num:4d}/{num_iterations}'
                f'  buf={len(deep_cfr.replay_buffer):>7d}'
                f'  adv={adv:.4f}{adv_arrow}'
                f'  strat={strat:.4f}{strat_arrow}'
                f'{fold_str}'
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

                # Spot validation — runs ~500 games to get enough postflop samples
                val_results = run_validation(
                    deep_cfr, engine, num_games=500, verbose=False
                )
                _last_trash_fold = val_results.get('trash_offsuit', {}).get('fold_pct')
                _last_72o_fold   = val_results.get('72o',           {}).get('fold_pct')
                print(f'  {validation_summary_line(val_results)}')
    finally:
        pool.terminate()
        pool.join()

    final = str(models_dir / 'final_model_enhanced.pt')
    deep_cfr.save_models(final)
    print(f'\nTraining complete. Model saved to {final}')
    engine.close()


if __name__ == '__main__':
    # Must be set before Pool spawns workers — children inherit this env var
    # and Python reads it at interpreter startup, before any imports run.
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    # 'spawn' is required when combining CUDA + multiprocessing
    mp.set_start_method('spawn', force=True)
    main()
