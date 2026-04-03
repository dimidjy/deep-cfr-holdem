"""
Conservative Q-Learning (CQL) from hand histories.

Offline RL approach: learns Q-values (expected reward per action) directly from
logged hand history data. Unlike behavioral cloning, CQL naturally learns when
to fold (negative Q-value actions) without needing synthetic fold labels.

The conservative regularizer penalizes Q-values on out-of-distribution actions,
keeping the policy close to the behavior policy while still improving.

Usage:
  # Train on all showdown players
  python3 scripts/train_cql.py --epochs 100 --output models/cql_v1.pt --no-cache

  # Train on good players only
  python3 scripts/train_cql.py --good-players data/good_players.txt --output models/cql_v1.pt --no-cache
"""
import sys
import os
import warnings
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import copy
import time
import json
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.training.deep_cfr import AdvantageNetwork, StrategyNetwork
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.parser import parse_file
from validate_spots import run_validation, validation_summary_line


# ──────────────────────────────────────────────────────────────────────────────
# Q-Network (reuses AdvantageNetwork architecture)
# ──────────────────────────────────────────────────────────────────────────────

class QNetwork(nn.Module):
    """Q-network: maps observation → Q-value for each action."""

    def __init__(self, obs_size: int, action_size: int,
                 hidden_layers: list, dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_size = obs_size
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.LeakyReLU(0.01))
            layers.append(nn.Dropout(dropout))
            if i + 1 < len(hidden_layers) and hidden_layers[i + 1] == h:
                from src.training.deep_cfr import ResidualBlock
                layers.append(ResidualBlock(h, dropout))
            prev_size = h
        layers.append(nn.Linear(prev_size, action_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ──────────────────────────────────────────────────────────────────────────────
# CQL-compatible wrapper for validation (mimics DeepCFR interface)
# ──────────────────────────────────────────────────────────────────────────────

class CQLModelWrapper:
    """Wraps a QNetwork to mimic the DeepCFR interface for run_validation."""

    def __init__(self, q_net, obs_size, action_size, num_players, device,
                 iteration=0, temperature=1.0, exclude_fold=False):
        self.action_size = action_size
        self.num_players = num_players
        self.obs_size = obs_size
        self.device = torch.device(device)
        self.iteration = iteration
        self.temperature = temperature
        self.exclude_fold = exclude_fold

        # Wrap Q-net as a StrategyNetwork-like object for validation
        self.q_net = q_net
        self.strategy_networks = [q_net] * num_players
        self.advantage_networks = [q_net] * num_players

    def get_strategy(self, observation: np.ndarray, player_idx: int,
                     legal_actions: list) -> np.ndarray:
        # Pad observation if model expects more dims
        if len(observation) < self.obs_size:
            padded = np.zeros(self.obs_size, dtype=np.float32)
            padded[:len(observation)] = observation
            if len(observation) >= 19 and self.obs_size > len(observation):
                pot_odds = observation[10]
                equity = observation[18]
                evo_idx = self.obs_size - 1
                padded[evo_idx] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
            observation = padded
        obs_t = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t).cpu().numpy()[0]

        # Handle 6-action model (split check/call)
        if self.action_size == 6:
            return self._strategy_6to5(q_values, observation, legal_actions)

        # Standard 5-action softmax
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        effective_legal = [a for a in legal_actions if not (self.exclude_fold and a == 0)] or [1]
        for a in effective_legal:
            masked[a] = q_values[a] / self.temperature
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        legal_mask = np.zeros(self.action_size, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy

    def _strategy_6to5(self, q_values: np.ndarray, observation: np.ndarray,
                       legal_actions: list) -> np.ndarray:
        """Convert 6-action Q-values to 5-action strategy for engine compatibility.

        CQL 6-action: 0=fold, 1=check, 2=call, 3=raise_half, 4=raise_pot, 5=allin
        Engine 5-action: 0=fold, 1=check/call, 2=raise_half, 3=raise_pot, 4=allin

        Based on context (facing bet or not), use Q(check) or Q(call) for action 1.
        """
        # Determine if facing a bet
        call_norm = float(observation[31]) if len(observation) > 31 else 0.0
        facing_bet = call_norm > 0.01

        # Map 6-action Q-values to the relevant ones for this context
        # CQL action → engine action: 0→0, 1→1(check), 2→1(call), 3→2, 4→3, 5→4
        mapped_q = np.full(5, -1e9, dtype=np.float32)
        mapped_q[0] = q_values[0]                              # fold
        mapped_q[1] = q_values[2] if facing_bet else q_values[1]  # call or check
        mapped_q[2] = q_values[3]                              # raise_half
        mapped_q[3] = q_values[4]                              # raise_pot
        mapped_q[4] = q_values[5]                              # all_in

        # Softmax over legal actions
        masked = np.full(5, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = mapped_q[a] / self.temperature
        masked -= masked.max()
        exp_s = np.exp(masked)
        strategy = exp_s / exp_s.sum()

        legal_mask = np.zeros(5, dtype=np.float32)
        legal_mask[legal_actions] = 1.0
        strategy *= legal_mask
        s = strategy.sum()
        if s > 0:
            strategy /= s
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        return strategy


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────────────

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
    )


# ──────────────────────────────────────────────────────────────────────────────
# Offline Replay Dataset — stores (s, a, r, s', done, legal, legal_next)
# ──────────────────────────────────────────────────────────────────────────────

class _IndexedReplayDataset(Dataset):
    """Zero-copy dataset that indexes into shared numpy arrays."""

    def __init__(self, states, actions, rewards, next_states, dones,
                 legal_masks, next_legal_masks, weights, indices):
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.next_states = next_states
        self.dones = dones
        self.legal_masks = legal_masks
        self.next_legal_masks = next_legal_masks
        self.weights = weights
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        return (torch.from_numpy(self.states[i]),
                torch.tensor(self.actions[i], dtype=torch.long),
                torch.tensor(self.rewards[i], dtype=torch.float32),
                torch.from_numpy(self.next_states[i]),
                torch.tensor(self.dones[i], dtype=torch.float32),
                torch.from_numpy(self.legal_masks[i]),
                torch.from_numpy(self.next_legal_masks[i]),
                torch.tensor(self.weights[i], dtype=torch.float32))


# ──────────────────────────────────────────────────────────────────────────────
# Data collection (reuses parse infrastructure from train_supervised)
# ──────────────────────────────────────────────────────────────────────────────

def _prefilter_files(hh_files, players, hh_dir):
    """Use ripgrep to quickly find files containing target player names."""
    import subprocess
    import shutil
    import re

    if not players:
        return hh_files

    rg_path = shutil.which('rg')
    if not rg_path:
        return hh_files

    matched_files = set()
    try:
        # Use -f - to pass patterns via stdin (avoids "Argument list too long")
        result = subprocess.run(
            [rg_path, '-l', '--no-messages', '-f', '-', str(hh_dir)],
            input='\n'.join(re.escape(p) for p in players),
            capture_output=True, text=True, timeout=1200
        )
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if line:
                matched_files.add(os.path.abspath(line))
    except Exception:
            return hh_files

    print(f'  Pre-filter: {len(matched_files)}/{len(hh_files)} files contain target players')
    return [f for f in hh_files if os.path.abspath(str(f)) in matched_files]


# Module-level worker config
_PARSE_CFG = {}


def _parse_worker_init(num_players, good_players, bad_players, postflop_only,
                       use_behavioral_stats=False):
    _PARSE_CFG['num_players'] = num_players
    _PARSE_CFG['good_players'] = good_players
    _PARSE_CFG['bad_players'] = bad_players
    _PARSE_CFG['postflop_only'] = postflop_only
    _PARSE_CFG['builder'] = ObservationBuilder(num_players=num_players)
    _PARSE_CFG['behavioral_stats'] = use_behavioral_stats


def _parse_single_file(file_path):
    """Parse one HH file — returns list of (hand_id, player_name, dps_for_player, net_dollars).

    For CQL we collect TWO types of data:
    1. Known-cards players (showdown): full obs with real equity, all their actions
    2. Fold-only transitions from ALL players: we parse with allow_unknown_cards=True
       but only keep the final fold action. Fold reward = -invested (known exactly).
       The unknown-card obs has approximate equity (~0.45) but CQL relies on reward
       signal, not features, so this is acceptable.
    """
    num_players = _PARSE_CFG['num_players']
    good_players = _PARSE_CFG['good_players']
    bad_players = _PARSE_CFG['bad_players']
    postflop_only = _PARSE_CFG['postflop_only']
    builder = _PARSE_CFG['builder']
    use_bh = _PARSE_CFG['behavioral_stats']

    file_results = []  # list of (hand_id, player_name, sorted_dps, net_dollars)
    file_total = 0
    file_used = 0

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return file_results, file_total, file_used

    for hand in hands:
        if abs(hand.big_blind_dollars - 0.05) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue

        file_total += 1
        if not hand.has_showdown:
            continue

        known_players = set(hand.shown_hands.keys())
        if good_players:
            target_known = known_players & good_players
        else:
            target_known = known_players

        # Also include bad players (their losing actions teach CQL what NOT to do)
        target_bad = (known_players & bad_players) if bad_players else set()

        if not target_known and not target_bad:
            continue

        # Compute player net profit (correct handling of raises)
        player_winnings = {}
        for w in hand.result.winners:
            name = w['name']
            player_winnings[name] = player_winnings.get(name, 0) + w['amount_dollars']

        player_invested = defaultdict(float)
        round_committed = defaultdict(float)
        current_street = 'preflop'
        for action in hand.actions:
            if action.street != current_street:
                current_street = action.street
                round_committed.clear()
            name = action.player_name
            if action.is_blind or action.action_type in ('bet', 'call'):
                player_invested[name] += action.amount_dollars
                round_committed[name] += action.amount_dollars
            elif action.action_type == 'raise':
                to_amt = action.amount_dollars
                already = round_committed.get(name, 0)
                delta = to_amt - already
                if delta > 0:
                    player_invested[name] += delta
                round_committed[name] = to_amt

        file_used += 1
        bb = hand.big_blind_dollars

        # ── 1) Known-cards players: full decision sequences ──────────────
        all_targets = target_known | target_bad
        dps = builder.build_decision_points(hand, allow_unknown_cards=False,
                                            source_file=str(file_path),
                                            behavioral_stats=use_bh)
        if dps:
            player_dps = defaultdict(list)
            for dp in dps:
                if postflop_only and dp.street == 'preflop':
                    continue
                if dp.player_name in all_targets:
                    player_dps[dp.player_name].append(dp)

            for player_name, pdps in player_dps.items():
                won = player_winnings.get(player_name, 0.0)
                invested = player_invested.get(player_name, 0.0)
                net = won - invested
                file_results.append((hand.hand_id, player_name, pdps, net))

        # ── 2) Synthetic fold transitions from showdown LOSERS ────────────
        # Two types of synthetic folds:
        #
        # A) EQUITY-BASED: equity < pot_odds_needed → call was mathematically
        #    wrong. These get full weight (the decision was clearly bad).
        #
        # B) LOSS-BASED: equity >= needed (cooler/unlucky), but the loss was
        #    large (>10 BB). These get reduced weight proportional to how much
        #    the fold would save. This teaches the model that even "correct"
        #    calls can be expensive, nudging it toward folding in big pots
        #    with marginal hands — without hard-overriding coolers.
        #
        # No unknown-cards folds (section 2 removed — garbage equity data).
        if dps:
            for player_name in known_players:
                won = player_winnings.get(player_name, 0.0)
                invested = player_invested.get(player_name, 0.0)
                net = won - invested
                if net >= 0:
                    continue  # winner, skip

                net_bb = net / bb

                player_postflop = [dp for dp in dps
                                   if dp.player_name == player_name
                                   and dp.street != 'preflop'
                                   and 0 in dp.legal_actions]
                if not player_postflop:
                    continue

                # Track cumulative investment by street
                invested_before = {}
                cum = 0.0
                prev_street = None
                for action in hand.actions:
                    if action.player_name != player_name:
                        continue
                    if action.street != prev_street:
                        invested_before[action.street] = cum
                        prev_street = action.street
                    if action.is_blind or action.action_type in ('bet', 'call', 'raise'):
                        cum += action.amount_dollars

                for dp in player_postflop:
                    obs = dp.observation
                    if len(obs) <= 18:
                        continue

                    inv_before = invested_before.get(dp.street, 0.0)
                    fold_net = -inv_before
                    saved_bb = (fold_net - net) / bb  # how much fold saves in BB

                    if saved_bb < 1.0:
                        continue  # fold doesn't save enough

                    pot_odds = float(obs[10])
                    equity = float(obs[18])
                    made_hand = float(obs[19]) if len(obs) > 19 else 0.0
                    equity_needed = 1.0 - pot_odds

                    # Never create fold labels for strong hands (two pair+)
                    if made_hand >= 0.25:
                        continue

                    # Create fold when equity is below what's needed to call
                    if equity < equity_needed:
                        fake_dp = copy.copy(dp)
                        fake_dp.actual_action = 0
                        file_results.append((hand.hand_id,
                                             player_name + '_sf',
                                             [fake_dp], fold_net))
                    # Also fold when loss is large (>5 BB saved)
                    elif saved_bb >= 5.0 and made_hand < 0.20:
                        fake_dp = copy.copy(dp)
                        fake_dp.actual_action = 0
                        file_results.append((hand.hand_id,
                                             player_name + '_sf',
                                             [fake_dp], fold_net))

    return file_results, file_total, file_used


# ──────────────────────────────────────────────────────────────────────────────
# Self-play data generation (teaches correct folding)
# ──────────────────────────────────────────────────────────────────────────────

def generate_selfplay_data(num_hands, num_players=6, model_fb='', model_ct='',
                           model_fb3='', target_obs_size=50):
    """Play hands using current best model, record all decisions with real rewards.

    This generates training data that showdown-only HH data is missing:
    - Players who correctly fold and save money (80% of real poker)
    - Full hand outcomes including folds, not just showdowns

    Returns: dict with same structure as collect_transitions output.
    """
    from src.engine import FastEnhancedPokerEngine
    from src.decision_engine import (
        DecisionEngine, DecisionConfig, SpotContext,
        IDX_POT_ODDS, IDX_EQUITY, NUM_ACTIONS,
    )

    eng = FastEnhancedPokerEngine(
        num_players=num_players, add_position_features=True,
        add_pot_odds_features=True, add_hand_strength_features=True,
        add_stack_features=True, add_equity_features=True,
        add_street_features=True, add_game_state_features=True,
        add_betting_history_features=True)

    # Load model for decision making
    dc = DecisionConfig(use_preflop_ranges=True, style_tuning=True,
                        strong_hand_guard=True, allin_filter=True,
                        raise_randomize=True)
    de = DecisionEngine(dc)
    if model_fb:
        de.load_models(facing_bet=model_fb, checked_to=model_ct,
                       facing_bet_3bet=model_fb3)
    de.load_preflop_ranges()

    engine_obs_size = eng.get_observation_space_size()  # 41
    action_size = 5
    bb_chips = 2.0  # engine uses chips, BB = 2 chips

    # Collect decision points: (obs, action, seat, hand_idx, street)
    all_decisions = []  # list of dicts per hand
    t0 = time.time()

    for hand_idx in range(num_hands):
        state = eng.reset()
        button = eng._button
        hand_decisions = []  # [(seat, obs_padded, action, street, legal_actions)]

        if hand_idx % 5000 == 0 and hand_idx > 0:
            elapsed = time.time() - t0
            print(f'  Self-play [{hand_idx}/{num_hands}] {hand_idx/elapsed:.0f} hands/s, '
                  f'{sum(len(d) for d in all_decisions)} decisions')

        while not eng.is_done():
            agent = eng.current_agent
            if agent is None:
                break
            legal = eng.get_legal_actions()
            if not legal:
                break

            seat = eng.agents.index(agent)
            stage = eng.get_stage()
            obs = eng.get_observation()
            cards = eng.get_player_cards(agent)
            rel_pos = (seat - button) % num_players

            # Pad obs to target size (add zeros for behavioral stats + equity_vs_odds)
            padded = np.zeros(target_obs_size, dtype=np.float32)
            padded[:len(obs)] = obs
            # Compute equity_vs_odds for the last dim
            if len(obs) >= 19 and target_obs_size > len(obs):
                pot_odds = obs[10] if len(obs) > 10 else 0
                equity = obs[18] if len(obs) > 18 else 0
                padded[target_obs_size - 1] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)

            # Skip preflop (handled by GTO ranges, not model)
            is_postflop = stage != 'Preflop'

            # Get action from DecisionEngine
            facing_bet = float(obs[10]) > 0.001 if len(obs) > 10 else False
            pot_bb = obs[8] * 600 if len(obs) > 8 else 0
            hole_cards = cards if stage == 'Preflop' else None

            ctx = SpotContext(
                obs=padded, legal_actions=legal, facing_bet=facing_bet,
                street=stage.lower() if stage != 'Preflop' else 'preflop',
                hole_cards=hole_cards, rel_pos=rel_pos, pot_bb=pot_bb,
                num_players=num_players,
                effective_stack=eng.get_player_chips(agent) / bb_chips)

            if de.has_models:
                action, _ = de.decide(ctx)
            else:
                # Random policy if no models
                action = int(np.random.choice(legal))

            # Record postflop decisions only
            if is_postflop:
                street_map = {'Flop': 1, 'Turn': 2, 'River': 3}
                hand_decisions.append({
                    'seat': seat,
                    'obs': padded.copy(),
                    'action': action,
                    'street': street_map.get(stage, 1),
                    'legal': legal,
                })

            eng.step(action)

        # Hand finished — get rewards
        rewards = eng.get_rewards()
        if hand_decisions:
            all_decisions.append((hand_decisions, rewards, eng.agents))

    elapsed = time.time() - t0
    n_decisions = sum(len(d) for d, _, _ in all_decisions)
    print(f'  Self-play: {num_hands} hands, {n_decisions} postflop decisions in {elapsed:.0f}s')

    # Convert to arrays matching collect_transitions format
    states = np.zeros((n_decisions, target_obs_size), dtype=np.float32)
    actions_arr = np.zeros(n_decisions, dtype=np.int64)
    rewards_arr = np.zeros(n_decisions, dtype=np.float32)
    next_states = np.zeros((n_decisions, target_obs_size), dtype=np.float32)
    dones_arr = np.zeros(n_decisions, dtype=np.float32)
    legal_masks = np.zeros((n_decisions, action_size), dtype=np.float32)
    next_legal_masks = np.zeros((n_decisions, action_size), dtype=np.float32)
    street_arr = np.zeros(n_decisions, dtype=np.int8)

    idx = 0
    for hand_decisions, rewards, agents in all_decisions:
        n = len(hand_decisions)
        for i, dec in enumerate(hand_decisions):
            seat = dec['seat']
            agent_name = agents[seat]
            reward_chips = rewards.get(agent_name, 0)
            net_bb = reward_chips / bb_chips

            is_last = (i + 1 >= n)
            is_fold = (dec['action'] == 0)

            states[idx] = dec['obs']
            actions_arr[idx] = dec['action']

            if is_last or is_fold:
                rewards_arr[idx] = np.clip(net_bb, -100, 100)
                dones_arr[idx] = 1.0
            else:
                rewards_arr[idx] = 0.0
                dones_arr[idx] = 0.0

            # Next state
            if not is_last and not is_fold and i + 1 < n:
                next_states[idx] = hand_decisions[i + 1]['obs']
                for a in hand_decisions[i + 1]['legal']:
                    if a < action_size:
                        next_legal_masks[idx, a] = 1.0
            else:
                next_states[idx] = dec['obs']
                for a in dec['legal']:
                    if a < action_size:
                        next_legal_masks[idx, a] = 1.0

            for a in dec['legal']:
                if a < action_size:
                    legal_masks[idx, a] = 1.0

            street_arr[idx] = dec['street']
            idx += 1

    states = states[:idx]
    actions_arr = actions_arr[:idx]
    rewards_arr = rewards_arr[:idx]
    next_states = next_states[:idx]
    dones_arr = dones_arr[:idx]
    legal_masks = legal_masks[:idx]
    next_legal_masks = next_legal_masks[:idx]
    street_arr = street_arr[:idx]

    # Stats
    n_folds = (actions_arr == 0).sum()
    n_calls = (actions_arr == 1).sum()
    n_raises = (actions_arr >= 2).sum()
    print(f'  Self-play data: {idx} transitions')
    print(f'    Folds: {n_folds} ({100*n_folds/idx:.1f}%)')
    print(f'    Calls: {n_calls} ({100*n_calls/idx:.1f}%)')
    print(f'    Raises: {n_raises} ({100*n_raises/idx:.1f}%)')
    print(f'    Rewards: mean={rewards_arr.mean():.2f}, std={rewards_arr.std():.2f}')

    return {
        'states': states, 'actions': actions_arr, 'rewards': rewards_arr,
        'next_states': next_states, 'dones': dones_arr,
        'legal_masks': legal_masks, 'next_legal_masks': next_legal_masks,
        'streets': street_arr, 'source_files': [], 'obs_size': target_obs_size,
    }


def collect_transitions(hh_dir, num_players, max_hands, good_players,
                        bad_players, no_cache, postflop_only,
                        use_behavioral_stats=False, workers=0):
    """Parse HH files and build offline RL transitions (s, a, r, s', done)."""
    hh_path = Path(hh_dir)
    all_hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'  Found {len(all_hh_files)} .txt files in {hh_dir}')

    all_target_players = (good_players or set()) | (bad_players or set())
    hh_files = _prefilter_files(all_hh_files, all_target_players, hh_dir)
    if not hh_files:
        return []

    # Cache
    player_key = '|'.join(sorted(good_players)) if good_players else 'all'
    bad_key = '|'.join(sorted(bad_players))[:100] if bad_players else 'none'
    bh_key = 'bh' if use_behavioral_stats else ''
    cache_key = f'cql2|{hh_dir}|{len(all_hh_files)}|flt{len(hh_files)}|np{num_players}|max{max_hands}|g{player_key}|b{bad_key}|pf{postflop_only}|{bh_key}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f'cql_{cache_hash}.pkl'

    npz_cache = str(cache_file).replace('.pkl', '.npz')
    if os.path.exists(npz_cache) and not no_cache:
        print(f'  Loading cached CQL data from {npz_cache}...')
        data = np.load(npz_cache)
        n = len(data['actions'])
        print(f'  Loaded {n} transitions')
        return {
            'states': data['states'], 'actions': data['actions'],
            'rewards': data['rewards'], 'next_states': data['next_states'],
            'dones': data['dones'], 'legal_masks': data['legal_masks'],
            'next_legal_masks': data['next_legal_masks'],
            'streets': data['streets'], 'source_files': [],
            'obs_size': data['states'].shape[1],
        }

    # Parallel parsing
    import multiprocessing as mp
    parse_workers = workers if workers > 0 else min(48, max(1, mp.cpu_count() - 1))

    # Limit files upfront to avoid pool.terminate() deadlock
    # Estimate ~50 showdown hands per file
    if max_hands > 0:
        max_files = max_hands // 50 + 200
        if len(hh_files) > max_files:
            print(f'  Limiting to {max_files} files (est. enough for {max_hands} hands)')
            hh_files = hh_files[:max_files]

    all_results = []
    total_hands = 0
    hands_used = 0
    t0 = time.time()

    print(f'  Parsing {len(hh_files)} files with {parse_workers} workers...')

    pool = mp.Pool(processes=parse_workers, initializer=_parse_worker_init,
                   initargs=(num_players, good_players, bad_players, postflop_only,
                             use_behavioral_stats))
    try:
        results = pool.imap_unordered(_parse_single_file, hh_files, chunksize=16)
        files_done = 0
        for file_results, file_total, file_used in results:
            all_results.extend(file_results)
            total_hands += file_total
            hands_used += file_used
            files_done += 1

            if files_done % 500 == 0:
                elapsed = time.time() - t0
                print(f'  [{files_done}/{len(hh_files)}] {total_hands} hands, '
                      f'{len(all_results)} player-hands ({elapsed:.0f}s)')

            if max_hands > 0 and hands_used >= max_hands:
                print(f'  Reached {max_hands} hands limit')
                break
    finally:
        pool.close()
        pool.join()

    elapsed = time.time() - t0
    print(f'  Parsed {files_done} files in {elapsed:.1f}s')
    print(f'  Player-hand sequences: {len(all_results)}')

    # Count total transitions for pre-allocation
    print(f'  Building transitions...')
    t_build = time.time()
    total_trans = sum(len(dps) for _, _, dps, _ in all_results if dps)

    obs_size = all_results[0][2][0].observation.shape[0] if all_results and all_results[0][2] else 41
    action_size = 5
    bb = 0.05

    # Pre-allocate numpy arrays
    states = np.zeros((total_trans, obs_size), dtype=np.float32)
    actions_arr = np.zeros(total_trans, dtype=np.int64)
    rewards_arr = np.zeros(total_trans, dtype=np.float32)
    next_states = np.zeros((total_trans, obs_size), dtype=np.float32)
    dones_arr = np.zeros(total_trans, dtype=np.float32)
    legal_masks = np.zeros((total_trans, action_size), dtype=np.float32)
    next_legal_masks = np.zeros((total_trans, action_size), dtype=np.float32)
    street_arr = np.zeros(total_trans, dtype=np.int8)
    source_files = []

    street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    idx = 0

    for hand_id, player_name, dps, net_dollars in all_results:
        n = len(dps)
        if n == 0:
            continue

        net_bb = net_dollars / bb

        for i, dp in enumerate(dps):
            is_terminal = (i + 1 >= n)
            reward = net_bb if is_terminal else 0.0

            if dp.actual_action == 0:  # fold = always terminal
                reward = net_bb
                is_terminal = True


            states[idx] = dp.observation
            actions_arr[idx] = dp.actual_action
            rewards_arr[idx] = np.clip(reward, -100, 100)
            dones_arr[idx] = 1.0 if is_terminal else 0.0

            if not is_terminal and i + 1 < n:
                next_states[idx] = dps[i + 1].observation
                for a in dps[i + 1].legal_actions:
                    next_legal_masks[idx, a] = 1.0
            else:
                next_states[idx] = dp.observation
                for a in dp.legal_actions:
                    next_legal_masks[idx, a] = 1.0

            for a in dp.legal_actions:
                legal_masks[idx, a] = 1.0

            street_arr[idx] = street_map.get(dp.street, 1)
            source_files.append(dp.source_file or '')
            idx += 1

    # Trim to actual size (some may have been skipped)
    states = states[:idx]
    actions_arr = actions_arr[:idx]
    rewards_arr = rewards_arr[:idx]
    next_states = next_states[:idx]
    dones_arr = dones_arr[:idx]
    legal_masks = legal_masks[:idx]
    next_legal_masks = next_legal_masks[:idx]
    street_arr = street_arr[:idx]

    print(f'  Total transitions: {idx} ({time.time() - t_build:.0f}s)')

    # Cache as compressed npz (much faster than pickle for numpy arrays)
    try:
        np.savez_compressed(
            str(cache_file).replace('.pkl', '.npz'),
            states=states, actions=actions_arr, rewards=rewards_arr,
            next_states=next_states, dones=dones_arr,
            legal_masks=legal_masks, next_legal_masks=next_legal_masks,
            streets=street_arr,
        )
        npz_path = str(cache_file).replace('.pkl', '.npz')
        mb = os.path.getsize(npz_path) / 1e6
        print(f'  Cached to {npz_path} ({mb:.0f} MB)')
    except Exception as e:
        print(f'  Cache save failed: {e}')

    return {
        'states': states, 'actions': actions_arr, 'rewards': rewards_arr,
        'next_states': next_states, 'dones': dones_arr,
        'legal_masks': legal_masks, 'next_legal_masks': next_legal_masks,
        'streets': street_arr, 'source_files': source_files,
        'obs_size': obs_size,
    }


def build_cql_dataset(trans_data, spot_filter=None, pot_filter=None):
    """Process pre-built arrays: add equity_vs_odds, scale rewards, compute weights.

    spot_filter: None (all), 'facing_bet' (call_amount > 0), 'checked_to' (call_amount == 0)
    """
    states = trans_data['states']
    actions = trans_data['actions']
    rewards = trans_data['rewards'].astype(np.float32).copy()
    next_states = trans_data['next_states']
    dones = trans_data['dones']
    legal_masks = trans_data['legal_masks']
    next_legal_masks = trans_data['next_legal_masks']
    streets = trans_data['streets']

    # ── Filter by spot type if requested ──
    if spot_filter and states.shape[1] > 10:
        # obs[10] = pot_odds = pot/(pot+call). pot_odds > 0 means facing a bet.
        pot_odds = states[:, 10]
        if spot_filter == 'facing_bet':
            mask = pot_odds > 0.001
        elif spot_filter == 'checked_to':
            mask = pot_odds <= 0.001
        else:
            mask = np.ones(len(states), dtype=bool)

        n_before = len(states)
        states = states[mask]
        actions = actions[mask]
        rewards = rewards[mask]
        next_states = next_states[mask]
        dones = dones[mask]
        legal_masks = legal_masks[mask]
        next_legal_masks = next_legal_masks[mask]
        streets = streets[mask]
        print(f'  Spot filter [{spot_filter}]: {n_before:,} → {len(states):,} '
              f'({100*len(states)/n_before:.1f}%)')

    # ── Filter by pot size (normal vs 3-bet pots) ──
    if pot_filter and states.shape[1] > 8:
        # obs[8] = pot_normalized = pot / (6 * 100)
        # 12BB = 12/600 = 0.02
        pot_norm = states[:, 8]
        _3BET_THRESHOLD = 0.02  # 12BB in normalized units
        if pot_filter == 'normal_pot':
            mask = pot_norm <= _3BET_THRESHOLD
        elif pot_filter == '3bet_pot':
            mask = pot_norm > _3BET_THRESHOLD
        else:
            mask = np.ones(len(states), dtype=bool)

        n_before = len(states)
        states = states[mask]
        actions = actions[mask]
        rewards = rewards[mask]
        next_states = next_states[mask]
        dones = dones[mask]
        legal_masks = legal_masks[mask]
        next_legal_masks = next_legal_masks[mask]
        streets = streets[mask]
        print(f'  Pot filter [{pot_filter}]: {n_before:,} → {len(states):,} '
              f'({100*len(states)/n_before:.1f}%)')

    if spot_filter:
        # For checked_to: mask out fold action (folding when checked to is never correct)
        # Keep all transitions but remap fold actions to check (action 1)
        # and remove fold from legal masks so CQL never considers it
        if spot_filter == 'checked_to':
            n_folds = (actions == 0).sum()
            actions[actions == 0] = 1  # remap fold → check
            legal_masks[:, 0] = 0.0
            next_legal_masks[:, 0] = 0.0
            print(f'  Remapped {n_folds:,} fold→check, removed fold from legal masks')

    # Street weights (vectorized)
    street_w = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    weights = street_w[streets]
    weights /= weights.mean()

    # Append equity_vs_odds feature
    obs_size = states.shape[1]
    if obs_size >= 19:
        evo_s = np.clip(states[:, 18] - (1.0 - states[:, 10]), -1.0, 1.0).reshape(-1, 1)
        evo_ns = np.clip(next_states[:, 18] - (1.0 - next_states[:, 10]), -1.0, 1.0).reshape(-1, 1)
        states = np.hstack([states, evo_s.astype(np.float32)])
        next_states = np.hstack([next_states, evo_ns.astype(np.float32)])
        print(f'  Added equity_vs_odds feature → obs_dim={states.shape[1]}')

    # Scale rewards (÷10 so typical ±50BB → ±5)
    rewards /= 10.0
    print(f'  Reward stats (scaled): mean={rewards.mean():.3f}, std={rewards.std():.3f}, '
          f'min={rewards.min():.1f}, max={rewards.max():.1f}')

    # ── Oversample folds to balance action distribution ──────────────
    # Showdown data is heavily biased toward calls. Oversample folds,
    # with extra weight on AIR folds (which are most underrepresented).
    fold_mask = actions == 0
    n_folds = fold_mask.sum()
    n_total = len(actions)

    if spot_filter == 'facing_bet':
        target_fold_pct = 0.45
    else:
        target_fold_pct = 0.167
    target_folds = int(target_fold_pct / (1.0 - target_fold_pct) * (n_total - n_folds))
    extra_folds = max(0, target_folds - n_folds)

    # For facing_bet: balance folds by hand strength.
    # Problem: too many pair/overpair folds → model folds strong hands.
    #          too few air folds → model calls with nothing.
    # Fix: downsample pair folds, oversample air folds.
    #
    # Tuning knobs (adjust these to fix specific issues):
    # v2: reduced air fold emphasis, model was folding too much with air/draws
    # Analysis showed 88% fold air vs winners' ~50% actual fold rate
    _FOLD_KEEP_AIR = 0.7       # keep 70% of air folds (was 1.0 — too many air folds)
    _FOLD_KEEP_DRAW = 0.6      # keep 60% of draw folds (was 0.8 — draws should call more)
    _FOLD_KEEP_PAIR = 0.4      # keep 40% of one-pair folds
    _FOLD_KEEP_STRONG = 0.1    # keep 10% of two pair+ folds
    _FOLD_OVERSAMPLE_AIR = 2.0 # oversample air folds 2x (was 4.0 — too aggressive)

    if spot_filter == 'facing_bet' and n_folds > 0:
        fold_indices = np.where(fold_mask)[0]
        made_hand = states[:, 19] if states.shape[1] > 19 else np.zeros(len(states))
        has_draw = np.zeros(len(states), dtype=bool)
        if states.shape[1] > 21:
            has_draw = (states[:, 20] > 0.5) | (states[:, 21] > 0.5)

        # Step 1: Downsample problematic folds
        rng_ds = np.random.RandomState(789)
        keep_mask = np.ones(len(states), dtype=bool)
        n_removed = {'air': 0, 'draw': 0, 'pair': 0, 'strong': 0}

        for idx in fold_indices:
            mh = made_hand[idx]
            draw = has_draw[idx]
            if mh < 0.01 and not draw:
                keep = rng_ds.random() < _FOLD_KEEP_AIR
                if not keep: n_removed['air'] += 1
            elif mh < 0.01 and draw:
                keep = rng_ds.random() < _FOLD_KEEP_DRAW
                if not keep: n_removed['draw'] += 1
            elif mh < 0.25:
                keep = rng_ds.random() < _FOLD_KEEP_PAIR
                if not keep: n_removed['pair'] += 1
            else:
                keep = rng_ds.random() < _FOLD_KEEP_STRONG
                if not keep: n_removed['strong'] += 1
            keep_mask[idx] = keep

        states = states[keep_mask]
        actions = actions[keep_mask]
        rewards = rewards[keep_mask]
        next_states = next_states[keep_mask]
        dones = dones[keep_mask]
        legal_masks = legal_masks[keep_mask]
        next_legal_masks = next_legal_masks[keep_mask]
        weights = weights[keep_mask]

        print(f'  Fold downsample: removed pair={n_removed["pair"]} '
              f'draw={n_removed["draw"]} strong={n_removed["strong"]} '
              f'air={n_removed["air"]}')

        # Recalculate fold stats after downsample
        fold_mask = actions == 0
        n_folds = fold_mask.sum()
        n_total = len(actions)

        # Step 2: Oversample remaining folds (mostly air now)
        target_folds = int(target_fold_pct / (1.0 - target_fold_pct) * (n_total - n_folds))
        extra_folds = max(0, target_folds - n_folds)

        if extra_folds > 0:
            fold_indices = np.where(fold_mask)[0]
            made_hand = states[:, 19] if states.shape[1] > 19 else np.zeros(len(states))

            # Weight: air folds get heavy oversample
            weights_os = np.zeros(len(fold_indices), dtype=np.float64)
            for i, fi in enumerate(fold_indices):
                mh = made_hand[fi]
                if mh < 0.01:
                    weights_os[i] = _FOLD_OVERSAMPLE_AIR
                elif mh < 0.25:
                    weights_os[i] = 1.0
                else:
                    weights_os[i] = 0.1
            weights_os /= weights_os.sum()

            rng_os = np.random.RandomState(123)
            oversample_idx = rng_os.choice(fold_indices, size=extra_folds,
                                            replace=True, p=weights_os)

            n_air_os = (made_hand[oversample_idx] < 0.01).sum()
            n_pair_os = ((made_hand[oversample_idx] >= 0.01) &
                         (made_hand[oversample_idx] < 0.25)).sum()
            print(f'  Fold oversample: air={n_air_os} pair={n_pair_os} '
                  f'strong={extra_folds - n_air_os - n_pair_os}')

            states = np.concatenate([states, states[oversample_idx]], axis=0)
            actions = np.concatenate([actions, actions[oversample_idx]])
            rewards = np.concatenate([rewards, rewards[oversample_idx]])
            next_states = np.concatenate([next_states, next_states[oversample_idx]], axis=0)
            dones = np.concatenate([dones, dones[oversample_idx]])
            legal_masks = np.concatenate([legal_masks, legal_masks[oversample_idx]], axis=0)
            next_legal_masks = np.concatenate([next_legal_masks, next_legal_masks[oversample_idx]], axis=0)
            weights = np.concatenate([weights, weights[oversample_idx]])

        new_fold_pct = (actions == 0).sum() / len(actions)
        print(f'  Fold final: {(actions==0).sum()}/{len(actions)} ({100*new_fold_pct:.1f}%)')
        extra_folds = 0  # already done

        # Step 3: Oversample RAISES for facing_bet (model never re-raises)
        # Analysis: model raises 0% with air/draws, 2% with two pair+
        # Winners raise 5-13% with air, 24% with two pair+
        _RAISE_FB_STRONG = 4.0   # two pair+: heavy re-raise oversample
        _RAISE_FB_PAIR = 2.0     # one pair: moderate
        _RAISE_FB_DRAW = 2.0     # draws: semi-bluff raise
        _RAISE_FB_AIR = 1.0      # air: bluff raise

        raise_mask_fb = (actions >= 2)
        n_raises_fb = raise_mask_fb.sum()
        # Target: winners raise ~15% when facing bets
        target_raise_fb = 0.15
        target_r = int(target_raise_fb / (1.0 - target_raise_fb) * (len(actions) - n_raises_fb))
        extra_r = max(0, target_r - n_raises_fb)

        if extra_r > 0 and n_raises_fb > 0:
            raise_idx_fb = np.where(raise_mask_fb)[0]
            made_hand = states[:, 19] if states.shape[1] > 19 else np.zeros(len(states))
            has_draw_fb = np.zeros(len(states), dtype=bool)
            if states.shape[1] > 21:
                has_draw_fb = (states[:, 20] > 0.5) | (states[:, 21] > 0.5)

            w_fb = np.zeros(len(raise_idx_fb), dtype=np.float64)
            for i, ri in enumerate(raise_idx_fb):
                mh = made_hand[ri]
                if mh >= 0.25:
                    w_fb[i] = _RAISE_FB_STRONG
                elif mh >= 0.01:
                    w_fb[i] = _RAISE_FB_PAIR
                elif has_draw_fb[ri]:
                    w_fb[i] = _RAISE_FB_DRAW
                else:
                    w_fb[i] = _RAISE_FB_AIR
            w_fb /= w_fb.sum()

            rng_rfb = np.random.RandomState(321)
            raise_os_fb = rng_rfb.choice(raise_idx_fb, size=extra_r,
                                          replace=True, p=w_fb)

            states = np.concatenate([states, states[raise_os_fb]], axis=0)
            actions = np.concatenate([actions, actions[raise_os_fb]])
            rewards = np.concatenate([rewards, rewards[raise_os_fb]])
            next_states = np.concatenate([next_states, next_states[raise_os_fb]], axis=0)
            dones = np.concatenate([dones, dones[raise_os_fb]])
            legal_masks = np.concatenate([legal_masks, legal_masks[raise_os_fb]], axis=0)
            next_legal_masks = np.concatenate([next_legal_masks, next_legal_masks[raise_os_fb]], axis=0)
            weights = np.concatenate([weights, weights[raise_os_fb]])

            new_raise_pct = (actions >= 2).sum() / len(actions)
            print(f'  Raise oversample (facing_bet): {n_raises_fb} → {(actions>=2).sum()} '
                  f'({100*n_raises_fb/n_total:.1f}% → {100*new_raise_pct:.1f}%)')

    # Skip fold oversampling for checked_to spots — oversample raises instead
    if spot_filter == 'checked_to':
        extra_folds = 0
        # Good players bet/raise 36% when checked to, data has ~27%.
        # Oversample raises weighted by hand strength:
        # Strong hands should bet more, air bets (bluffs) less.
        #
        # Tuning knobs:
        # v2: increased draw/air to add more semi-bluffs and bluffs
        # Analysis: model bets 4% with air (winners: 14%), 14% with draws (winners: 25%)
        _RAISE_OS_STRONG = 4.0   # trips+: heavy oversample (should bet ~70%)
        _RAISE_OS_TWO_PAIR = 3.0 # two pair: strong oversample
        _RAISE_OS_PAIR = 2.0     # one pair: moderate (value bet sometimes)
        _RAISE_OS_DRAW = 2.5     # draws: semi-bluff more (was 1.5)
        _RAISE_OS_AIR = 1.5      # air: bluff more often (was 0.5)

        raise_mask = (actions >= 2)
        n_raises = raise_mask.sum()
        target_raise_pct = 0.36
        target_raises = int(target_raise_pct / (1.0 - target_raise_pct) * (n_total - n_raises))
        extra_raises = max(0, target_raises - n_raises)
        if extra_raises > 0 and n_raises > 0:
            raise_indices = np.where(raise_mask)[0]
            made_hand = states[:, 19] if states.shape[1] > 19 else np.zeros(len(states))
            has_draw = np.zeros(len(states), dtype=bool)
            if states.shape[1] > 21:
                has_draw = (states[:, 20] > 0.5) | (states[:, 21] > 0.5)

            weights_rs = np.zeros(len(raise_indices), dtype=np.float64)
            for i, ri in enumerate(raise_indices):
                mh = made_hand[ri]
                if mh >= 0.375:
                    weights_rs[i] = _RAISE_OS_STRONG
                elif mh >= 0.25:
                    weights_rs[i] = _RAISE_OS_TWO_PAIR
                elif mh >= 0.01:
                    weights_rs[i] = _RAISE_OS_PAIR
                elif has_draw[ri]:
                    weights_rs[i] = _RAISE_OS_DRAW
                else:
                    weights_rs[i] = _RAISE_OS_AIR
            weights_rs /= weights_rs.sum()

            rng_rs = np.random.RandomState(456)
            raise_os_idx = rng_rs.choice(raise_indices, size=extra_raises,
                                          replace=True, p=weights_rs)
            n_strong = (made_hand[raise_os_idx] >= 0.375).sum()
            n_pair = ((made_hand[raise_os_idx] >= 0.01) &
                      (made_hand[raise_os_idx] < 0.25)).sum()
            print(f'  Raise oversample: strong={n_strong} pair={n_pair} '
                  f'air={extra_raises - n_strong - n_pair}')
            states = np.concatenate([states, states[raise_os_idx]], axis=0)
            actions = np.concatenate([actions, actions[raise_os_idx]])
            rewards = np.concatenate([rewards, rewards[raise_os_idx]])
            next_states = np.concatenate([next_states, next_states[raise_os_idx]], axis=0)
            dones = np.concatenate([dones, dones[raise_os_idx]])
            legal_masks = np.concatenate([legal_masks, legal_masks[raise_os_idx]], axis=0)
            next_legal_masks = np.concatenate([next_legal_masks, next_legal_masks[raise_os_idx]], axis=0)
            weights = np.concatenate([weights, weights[raise_os_idx]])
            new_raise_pct = (actions >= 2).sum() / len(actions)
            print(f'  Raise oversampling: {n_raises} → {(actions>=2).sum()} '
                  f'({100*n_raises/n_total:.1f}% → {100*new_raise_pct:.1f}%)')

    if extra_folds > 0 and n_folds > 0:
        fold_indices = np.where(fold_mask)[0]
        # Sample with replacement from existing folds
        rng_os = np.random.RandomState(123)
        oversample_idx = rng_os.choice(fold_indices, size=extra_folds, replace=True)

        states = np.concatenate([states, states[oversample_idx]], axis=0)
        actions = np.concatenate([actions, actions[oversample_idx]])
        rewards = np.concatenate([rewards, rewards[oversample_idx]])
        next_states = np.concatenate([next_states, next_states[oversample_idx]], axis=0)
        dones = np.concatenate([dones, dones[oversample_idx]])
        legal_masks = np.concatenate([legal_masks, legal_masks[oversample_idx]], axis=0)
        next_legal_masks = np.concatenate([next_legal_masks, next_legal_masks[oversample_idx]], axis=0)
        weights = np.concatenate([weights, weights[oversample_idx]])

        new_fold_pct = (actions == 0).sum() / len(actions)
        print(f'  Fold oversampling: {n_folds} → {(actions==0).sum()} '
              f'({100*n_folds/n_total:.1f}% → {100*new_fold_pct:.1f}%)')

    return states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights


# ──────────────────────────────────────────────────────────────────────────────
# CQL Training
# ──────────────────────────────────────────────────────────────────────────────

def cql_train_epoch(q_net, q_target, dataloader, optimizer, device, scaler,
                    alpha_cql=1.0, gamma=0.99):
    """One epoch of CQL training. Returns avg loss components."""
    q_net.train()
    total_td_loss = 0.0
    total_cql_loss = 0.0
    total_loss = 0.0
    total_q_mean = 0.0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            # Current Q-values
            q_values = q_net(obs)  # (B, 5)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)  # (B,)

            # Double Q-learning: online net selects action, target net evaluates
            # Reduces overestimation bias vs standard max-Q
            with torch.no_grad():
                # Online net picks the best legal action
                next_q_online = q_net(next_obs)
                next_q_online_masked = next_q_online + (1.0 - next_legal) * (-1e9)
                best_actions = next_q_online_masked.argmax(dim=1, keepdim=True)
                # Target net evaluates that action
                next_q_target = q_target(next_obs)
                v_next = next_q_target.gather(1, best_actions).squeeze(1)
                # Bellman target
                target = rews + gamma * v_next * (1.0 - done)

            # TD loss (weighted)
            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            # Conservative regularizer:
            # Penalize Q-values for being high on all actions (logsumexp)
            # minus Q-value of the action actually taken (from data)
            # This pushes down Q on unseen actions, keeps Q on seen actions
            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)  # (B,)
            cql_loss = (logsumexp_q - q_taken).mean()

            loss = td_loss + alpha_cql * cql_loss

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()

        bs = obs.size(0)
        total_td_loss += td_loss.item() * bs
        total_cql_loss += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'td_loss': total_td_loss / total_samples,
        'cql_loss': total_cql_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
    }


@torch.no_grad()
def cql_eval_epoch(q_net, q_target, dataloader, device, alpha_cql=1.0, gamma=0.99):
    """Evaluate CQL on validation set."""
    q_net.eval()
    total_td_loss = 0.0
    total_cql_loss = 0.0
    total_loss = 0.0
    total_q_mean = 0.0
    correct = 0
    total_samples = 0

    for batch in dataloader:
        obs, acts, rews, next_obs, done, legal, next_legal, wts = [
            b.to(device, non_blocking=True) for b in batch]

        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            q_values = q_net(obs)
            q_taken = q_values.gather(1, acts.unsqueeze(1)).squeeze(1)

            # Double Q-learning (same as train)
            next_q_online = q_net(next_obs)
            next_q_online_masked = next_q_online + (1.0 - next_legal) * (-1e9)
            best_actions = next_q_online_masked.argmax(dim=1, keepdim=True)
            next_q_target = q_target(next_obs)
            v_next = next_q_target.gather(1, best_actions).squeeze(1)
            target = rews + gamma * v_next * (1.0 - done)

            td_loss = (wts * F.smooth_l1_loss(q_taken, target, reduction='none')).mean()

            q_legal = q_values + (1.0 - legal) * (-1e9)
            logsumexp_q = torch.logsumexp(q_legal, dim=1)
            cql_loss = (logsumexp_q - q_taken).mean()

            loss = td_loss + alpha_cql * cql_loss

        # Smart accuracy: reward-aware
        # - Fold actions: always match (folds have negative reward by
        #   definition since you lose invested chips, but they're correct)
        # - Non-fold with reward >= 0: matching = correct (winning action)
        # - Non-fold with reward < -0.5 (lost >5BB): NOT matching = correct
        #   (model should deviate from bad calls/raises that lost big)
        # - Non-fold with small loss (>-0.5): matching = correct (marginal)
        q_masked = q_values + (1.0 - legal) * (-1e9)
        preds = q_masked.argmax(dim=-1)
        matches = (preds == acts)
        is_fold = (acts == 0)
        big_loss = (rews < -0.5) & ~is_fold  # lost >5BB, not a fold
        # Correct = match folds + match non-big-loss + deviate from big loss
        correct += (matches & ~big_loss).sum().item()
        correct += (~matches & big_loss).sum().item()

        bs = obs.size(0)
        total_td_loss += td_loss.item() * bs
        total_cql_loss += cql_loss.item() * bs
        total_loss += loss.item() * bs
        total_q_mean += q_taken.mean().item() * bs
        total_samples += bs

    return {
        'td_loss': total_td_loss / total_samples,
        'cql_loss': total_cql_loss / total_samples,
        'total_loss': total_loss / total_samples,
        'q_mean': total_q_mean / total_samples,
        'accuracy': correct / total_samples,
    }


def soft_update(target_net, online_net, tau=0.005):
    """Polyak averaging: target ← τ * online + (1 - τ) * target."""
    for tp, op in zip(target_net.parameters(), online_net.parameters()):
        tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='CQL (Conservative Q-Learning) from hand histories')

    # Data
    parser.add_argument('--config', type=str, default='',
                        help='Path to supervised_config.json (loads data/training settings)')
    parser.add_argument('--hh-dir', type=str, default='data/poker_archive/')
    parser.add_argument('--good-players', type=str, default='')
    parser.add_argument('--bad-players', type=str, default='')
    parser.add_argument('--max-hands', type=int, default=0)
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of parsing workers (0=auto)')
    parser.add_argument('--behavioral-stats', action='store_true',
                        help='Append 8-dim behavioral stats to observations')
    parser.add_argument('--training-data', type=str, default='',
                        help='Pre-built CQL data (.npz from --build-data). Skips parsing.')
    parser.add_argument('--build-data', type=str, default='',
                        help='Build CQL data and save to this path, then exit. '
                             'Use with --training-data to skip parsing on next run.')
    parser.add_argument('--selfplay-hands', type=int, default=0,
                        help='Generate this many self-play hands and mix into training data. '
                             'Adds fold examples missing from showdown-only data.')
    parser.add_argument('--selfplay-model-fb', type=str,
                        default='models/cql_6p_facing_bet_v8.pt',
                        help='Facing-bet model for self-play generation')
    parser.add_argument('--selfplay-model-ct', type=str,
                        default='models/cql_6p_checked_to_v4.pt',
                        help='Checked-to model for self-play generation')
    parser.add_argument('--spot-filter', type=str, default='',
                        choices=['', 'facing_bet', 'checked_to'],
                        help='Train only on specific spot type: facing_bet or checked_to')
    parser.add_argument('--pot-filter', type=str, default='',
                        choices=['', 'normal_pot', '3bet_pot'],
                        help='Train only on pot size: normal_pot (<=12BB) or 3bet_pot (>12BB)')

    # CQL hyperparameters
    parser.add_argument('--alpha-cql', type=float, default=0.4,
                        help='CQL conservative regularizer (0.4=balanced, lower=more deviation from data)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                        help='Target network soft update rate')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--hidden-layers', type=str, default='512,512,256')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--val-split', type=float, default=0.1)
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Softmax temperature for policy extraction (lower=greedier)')

    # Validation
    parser.add_argument('--val-interval', type=int, default=10)
    parser.add_argument('--val-games', type=int, default=500)

    # Output
    parser.add_argument('--output', type=str, default='models/cql_model.pt')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint (loads weights + optimizer state)')
    parser.add_argument('--device', type=str, default=None)

    args = parser.parse_args()

    # Load config if provided
    if args.config:
        import json as _json
        with open(args.config) as _f:
            _cfg = _json.load(_f)
        _data = _cfg.get('data', {})
        _train = _cfg.get('train', {})
        if not args.hh_dir or args.hh_dir == 'data/poker_archive/':
            args.hh_dir = _data.get('hh_dir', args.hh_dir)
        if not args.good_players:
            args.good_players = _data.get('good_players', '')
        if not args.bad_players:
            args.bad_players = _data.get('bad_players', '')
        if not args.behavioral_stats:
            args.behavioral_stats = _train.get('behavioral_stats', False)
        if args.hidden_layers == '512,512,256':
            args.hidden_layers = ','.join(
                str(x) for x in _train.get('hidden_layers', [512, 512, 256]))
        if not hasattr(args, 'workers') or args.workers == 0:
            args.workers = _data.get('workers', 0)
        for key in ['epochs', 'lr', 'batch_size', 'dropout',
                     'alpha_cql', 'gamma', 'tau', 'temperature']:
            if key in _train and getattr(args, key.replace('-', '_'), None) is None:
                pass  # CLI takes precedence
            elif key in _train:
                cli_default = parser.get_default(key.replace('_', '-') if '-' in key else key)
                if getattr(args, key.replace('-', '_')) == cli_default:
                    setattr(args, key.replace('-', '_'), _train[key])
        if _train.get('output') and args.output == 'models/cql_model.pt':
            args.output = _train['output']

    postflop_only = True

    # Load player lists
    def _load_player_list(path):
        if not path:
            return set()
        p = Path(path)
        if not p.exists():
            print(f'Player file not found: {p}')
            return set()
        return set(line.strip() for line in p.read_text().splitlines()
                   if line.strip() and not line.strip().startswith('#'))

    good_players = _load_player_list(args.good_players)
    bad_players = _load_player_list(args.bad_players)
    if good_players:
        print(f'Good players ({len(good_players)}): loaded')
    if bad_players:
        print(f'Bad players ({len(bad_players)}): loaded')

    use_bh = bool(args.behavioral_stats)
    if use_bh:
        from src.hand_history import BEHAVIORAL_STATS_DIM
        print(f'Behavioral stats: ENABLED (+{BEHAVIORAL_STATS_DIM} dims)')

    num_players = 6
    action_size = 5
    hidden_layers = [int(x) for x in args.hidden_layers.split(',')]
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))

    # ── Header ───────────────────────────────────────────────────────────
    print('=' * 60)
    print('CQL  —  CONSERVATIVE Q-LEARNING')
    print('=' * 60)
    print(f'  device     : {device}')
    if device.type == 'cuda':
        print(f'  GPU        : {torch.cuda.get_device_name(0)}')
    print(f'  players    : {num_players}')
    print(f'  epochs     : {args.epochs}')
    print(f'  batch      : {args.batch_size}')
    print(f'  lr         : {args.lr}')
    print(f'  layers     : {hidden_layers}')
    print(f'  alpha_cql  : {args.alpha_cql}')
    print(f'  gamma      : {args.gamma}')
    print(f'  tau        : {args.tau}')
    print(f'  temperature: {args.temperature}')
    print(f'  behavioral : {use_bh}')
    print(f'  bad players: {len(bad_players)}')
    print('=' * 60)

    # ── Collect data ─────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  COLLECTING TRAINING DATA')
    print(f'{"="*60}')

    if args.training_data:
        # Load pre-built CQL data (skips parsing entirely)
        print(f'  Loading pre-built data from {args.training_data}...')
        t0 = time.time()
        data = np.load(args.training_data)
        trans_data = {
            'states': data['states'], 'actions': data['actions'],
            'rewards': data['rewards'], 'next_states': data['next_states'],
            'dones': data['dones'], 'legal_masks': data['legal_masks'],
            'next_legal_masks': data['next_legal_masks'],
            'streets': data['streets'], 'source_files': [],
            'obs_size': data['states'].shape[1],
        }
        print(f'  Loaded {len(trans_data["actions"])} transitions ({time.time()-t0:.0f}s)')
    else:
        transitions = collect_transitions(
            args.hh_dir, num_players, args.max_hands,
            good_players, bad_players, args.no_cache, postflop_only,
            use_behavioral_stats=use_bh, workers=args.workers)
        trans_data = transitions

        # Save data if --build-data specified
        if args.build_data:
            out = Path(args.build_data)
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f'  Saving CQL data to {out}...')
            np.savez_compressed(str(out),
                states=trans_data['states'], actions=trans_data['actions'],
                rewards=trans_data['rewards'], next_states=trans_data['next_states'],
                dones=trans_data['dones'], legal_masks=trans_data['legal_masks'],
                next_legal_masks=trans_data['next_legal_masks'],
                streets=trans_data['streets'])
            mb = out.stat().st_size / 1e6
            print(f'  Saved ({mb:.0f} MB). Use --training-data {out} to skip parsing.')
            if not args.epochs:
                return  # build-data only mode

    n_trans = len(trans_data['actions'])

    if n_trans < 100:
        print(f'Not enough data: {n_trans} transitions.')
        sys.exit(1)

    # ── Mix in self-play data (teaches correct folding) ──────────────────
    if args.selfplay_hands > 0:
        print(f'\n{"="*60}')
        print(f'  GENERATING SELF-PLAY DATA ({args.selfplay_hands:,} hands)')
        print(f'{"="*60}')
        sp_data = generate_selfplay_data(
            num_hands=args.selfplay_hands,
            model_fb=args.selfplay_model_fb,
            model_ct=args.selfplay_model_ct,
            model_fb3='',
            target_obs_size=trans_data['obs_size'],
        )
        # Merge: concatenate self-play transitions into HH transitions
        for key in ['states', 'actions', 'rewards', 'next_states', 'dones',
                    'legal_masks', 'next_legal_masks', 'streets']:
            trans_data[key] = np.concatenate([trans_data[key], sp_data[key]], axis=0)
        n_sp = len(sp_data['actions'])
        n_total = len(trans_data['actions'])
        print(f'  Mixed: {n_trans:,} HH + {n_sp:,} self-play = {n_total:,} total '
              f'({100*n_sp/n_total:.1f}% self-play)')
        n_trans = n_total

    # ── Build dataset ────────────────────────────────────────────────────
    print(f'Building CQL dataset...')
    obs_size = trans_data['obs_size']
    arrays = build_cql_dataset(trans_data, spot_filter=args.spot_filter or None,
                               pot_filter=args.pot_filter or None)
    states, actions, rewards, next_states, dones, legal_masks, next_legal_masks, weights = arrays
    obs_size = states.shape[1]  # may have grown with equity_vs_odds

    # Stats
    print(f'  Observation size: {obs_size}')
    print(f'  Total transitions: {len(states)}')

    action_names = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
    print(f'  Action distribution:')
    for a in range(action_size):
        c = int((trans_data['actions'] == a).sum())
        print(f'    {action_names[a]:<12} {c:>8}  ({c/n_trans*100:.1f}%)')

    street_names = {0: 'preflop', 1: 'flop', 2: 'turn', 3: 'river'}
    print(f'  Street distribution:')
    for s in range(4):
        c = int((trans_data['streets'] == s).sum())
        if c > 0:
            print(f'    {street_names[s]:<12} {c:>8}  ({c/n_trans*100:.1f}%)')

    # ── Random train/val split (fast, no file-based grouping needed) ────
    print(f'  Splitting train/val...')
    n = len(states)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n)
    val_n = int(n * args.val_split)
    val_idx = perm[:val_n]
    train_idx = perm[val_n:]

    # Convert to tensors and use DataLoader with pin_memory
    print(f'  Converting to tensors...')
    t_gpu = time.time()

    train_dataset = _IndexedReplayDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, train_idx)
    val_dataset = _IndexedReplayDataset(
        states, actions, rewards, next_states, dones,
        legal_masks, next_legal_masks, weights, val_idx)

    train_n = len(train_idx)
    val_n_count = len(val_idx)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=24, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=24, pin_memory=True,
                            persistent_workers=True, prefetch_factor=4)

    print(f'  Train: {train_n}, Val: {val_n_count}')

    # ── Model ────────────────────────────────────────────────────────────
    q_net = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target = QNetwork(obs_size, action_size, hidden_layers, args.dropout).to(device)
    q_target.load_state_dict(q_net.state_dict())
    q_target.eval()

    total_params = sum(p.numel() for p in q_net.parameters())
    print(f'  Q-network params: {total_params:,}')

    optimizer = torch.optim.AdamW(q_net.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    start_epoch = 0
    best_val_loss = 0.0  # tracks best val_acc (higher = better)
    _val_acc_history = []  # smoothing window for noisy val_acc
    best_epoch = 0

    # Resume from checkpoint
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device)
        q_net.load_state_dict(resume_ckpt['model_state_dict'])
        if 'target_state_dict' in resume_ckpt:
            q_target.load_state_dict(resume_ckpt['target_state_dict'])
        else:
            q_target.load_state_dict(resume_ckpt['model_state_dict'])
        start_epoch = resume_ckpt.get('epoch', 0)
        print(f'  Resumed from {args.resume} (epoch {start_epoch})')
        # Step scheduler to correct position
        for _ in range(start_epoch):
            scheduler.step()

    # Validation engine + preflop ranges
    engine = _make_engine(num_players)
    from src.preflop_ranges import PreflopRanges
    preflop_ranges = PreflopRanges()
    print(f'  Preflop ranges loaded (GTO locked)')

    # ── Training loop ────────────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'  TRAINING CQL')
    print(f'{"="*60}')

    patience = 50
    patience_counter = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Random alpha: sample from range each epoch ──
    # Different conservatism each epoch explores the alpha space naturally.
    # The best checkpoint (by val_acc) selects the effective best alpha.
    _ALPHA_MIN = 0.33
    _ALPHA_MAX = 0.43
    _alpha_rng = np.random.RandomState(42)
    print(f'  Alpha range: [{_ALPHA_MIN:.2f}, {_ALPHA_MAX:.2f}] (random per epoch)')

    for epoch in range(start_epoch + 1, args.epochs + 1):
        t0 = time.time()

        current_alpha = round(_alpha_rng.uniform(_ALPHA_MIN, _ALPHA_MAX), 3)

        train_metrics = cql_train_epoch(
            q_net, q_target, train_loader, optimizer, device, scaler,
            current_alpha, args.gamma)

        # Soft update target network
        soft_update(q_target, q_net, args.tau)

        val_metrics = cql_eval_epoch(
            q_net, q_target, val_loader, device, current_alpha, args.gamma)

        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Track best (use smoothed val_acc to reduce noise from random alpha)
        improved = ''
        val_acc = val_metrics['accuracy']
        _val_acc_history.append(val_acc)
        # Use average of last 5 epochs as the smoothed metric
        _smooth_window = min(5, len(_val_acc_history))
        smoothed_acc = sum(_val_acc_history[-_smooth_window:]) / _smooth_window
        if smoothed_acc > best_val_loss:
            best_val_loss = smoothed_acc
            best_epoch = epoch
            patience_counter = 0
            improved = ' *best*'

            save_dict = {
                'model_state_dict': q_net.state_dict(),
                'target_state_dict': q_target.state_dict(),
                'obs_size': obs_size,
                'action_size': action_size,
                'hidden_layers': hidden_layers,
                'dropout': args.dropout,
                'num_players': num_players,
                'epoch': epoch,
                'alpha_cql': current_alpha,
                'gamma': args.gamma,
                'temperature': args.temperature,
                'model_type': 'cql',
                'behavioral_stats': use_bh,
            }
            torch.save(save_dict, str(output_path))
        else:
            patience_counter += 1

        print(f'  Epoch {epoch:>3}/{args.epochs}  '
              f'td={train_metrics["td_loss"]:.4f}  '
              f'cql={train_metrics["cql_loss"]:.4f}  '
              f'Q={train_metrics["q_mean"]:.3f}  '
              f'val_td={val_metrics["td_loss"]:.4f}  '
              f'val_acc={val_acc:.1%}({smoothed_acc:.1%})  '
              f'α={current_alpha:.3f}  lr={lr:.6f}  {elapsed:.1f}s{improved}')

        # Spot validation
        if args.val_interval > 0 and epoch % args.val_interval == 0:
            _excl_fold = (args.spot_filter == 'checked_to')
            wrapper = CQLModelWrapper(
                q_net, obs_size, action_size, num_players, device,
                iteration=epoch, temperature=args.temperature,
                exclude_fold=_excl_fold)
            val_results = run_validation(wrapper, engine, num_games=args.val_games,
                                        verbose=False, preflop_ranges=preflop_ranges)
            print(f'  {validation_summary_line(val_results)}')

        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}')
            break

    # ── Final validation ─────────────────────────────────────────────────
    best_ckpt = torch.load(str(output_path), map_location=device)
    q_net.load_state_dict(best_ckpt['model_state_dict'])
    _excl_fold = (args.spot_filter == 'checked_to')
    wrapper = CQLModelWrapper(
        q_net, obs_size, action_size, num_players, device,
        iteration=best_epoch, temperature=args.temperature,
        exclude_fold=_excl_fold)
    final_results = run_validation(wrapper, engine, num_games=args.val_games,
                                   verbose=False, preflop_ranges=preflop_ranges)

    print(f'\n{"="*60}')
    print(f'  CQL TRAINING COMPLETE')
    print(f'{"="*60}')
    print(f'  Best epoch:     {best_epoch}')
    print(f'  Best val loss:  {best_val_loss:.4f}')
    print(f'  Model saved to: {output_path}')
    summary = final_results.get('_summary', {})
    print(f'  Validation:     {summary.get("passed",0)}/{summary.get("total",0)} spots passed')
    print(f'  {validation_summary_line(final_results)}')
    print(f'{"="*60}')


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore'
    warnings.filterwarnings('ignore')
    logging.disable(logging.WARNING)
    main()
