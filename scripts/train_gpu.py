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


def _worker_init(shared_adv, shared_strat):
    """Store shared model references and suppress noisy warnings."""
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS
    _SHARED_ADV_NETS   = shared_adv
    _SHARED_STRAT_NETS = shared_strat
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
    global _SHARED_ADV_NETS, _SHARED_STRAT_NETS
    player_idx, seed, num_players = args
    np.random.seed(seed)

    from src.engine import EnhancedPokerEngine  # local import for subprocess
    engine = EnhancedPokerEngine(
        num_players=num_players,
        add_position_features=True,
        add_pot_odds_features=True,
        add_hand_strength_features=True,
        add_stack_features=True,
    )
    engine.reset()

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
            # Opponent: use current strategy network (self-play MCCFR)
            if _SHARED_STRAT_NETS is not None:
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                    strat = _SHARED_STRAT_NETS[agent_idx](obs_t).numpy()[0]
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
        _, final_reward, _ = episodes[ep_idx]
        final_reward = final_reward / _REWARD_SCALE  # normalise to ±1 range
        adv = all_advantages[entry_idx]

        mask = np.zeros(deep_cfr.action_size, dtype=np.float32)
        mask[legal] = 1.0
        adv = adv * mask

        pos = np.maximum(adv, 0.0) * mask
        pos_sum = pos.sum()
        strategy = pos / pos_sum if pos_sum > 0 else mask / mask.sum()

        ev = float(np.sum(strategy[legal] * adv[legal]))

        for a in legal:
            regret = (final_reward - ev) if a == action else (float(adv[a]) - ev)
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
                initargs=(shared_adv_nets, shared_strat_nets))
    print('Workers ready.\n')

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
            print(
                f'Iter {iter_num:4d}/{num_iterations}'
                f'  buf={len(deep_cfr.replay_buffer):>7d}'
                f'  adv={adv:.4f}'
                f'  strat={strat:.4f}'
                f'  trav={t_trav:.1f}s'
                f'  regret={t_regret:.1f}s'
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
