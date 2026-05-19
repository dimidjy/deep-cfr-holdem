"""
Validate trained Deep CFR model against real NL5 hand history data.

At every decision point from real PokerStars games, queries the model
for its strategy and compares to what the actual NL5 player did.

Usage:
  python3 scripts/validate_vs_hh.py --model models/checkpoint_iter_500.pt
  python3 scripts/validate_vs_hh.py --model models/final_model_enhanced.pt --hh-dir data/poker_archive/
  python3 scripts/validate_vs_hh.py --model models/final_model_enhanced.pt --strategy current
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import hashlib
import pickle
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

from src.engine import EnhancedPokerEngine
from src.training import DeepCFR
from src.training.deep_cfr import StrategyNetwork
from src.utils import load_config
from src.blended_model import BlendedModel
from src.hand_history import parse_directory, ObservationBuilder, DecisionPoint, BEHAVIORAL_STATS_DIM
from src.search import PlaytimeStrategy
from src.preflop_ranges import PreflopRanges

# Import classifiers and GTO targets from validate_spots
from validate_spots import (
    classify_preflop, classify_postflop,
    SPOTS, _spot_passes, _compute_spot_stats,
    FOLD, CALL, RAISE_HALF, RAISE_FULL, ALL_IN, AGGRESSIVE,
    GTO_OPEN, GTO_VPIP, GTO_3BET,
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants for EV / counterfactual analysis
# ──────────────────────────────────────────────────────────────────────────────
# NL5 big-blind size (used to convert dollar EV to BB).
_BB_DOLLARS = 0.05
# Default fold equity assumption when model raises where the NL5 player did not.
# Calibrated to NL5 average fold-to-CBet (~37%) and fold-to-3bet (~50-65%) blends.
_DEFAULT_P_FOLD = 0.30
# Approximate raise size as a fraction of pot, by action code.
# Raise½ ≈ 0.5×pot; RaisePot ≈ 1.0×pot; AllIn ≈ 2×pot (conservative).
_RAISE_SIZE_MULT = {RAISE_HALF: 0.5, RAISE_FULL: 1.0, ALL_IN: 2.0}
# When opponent calls the model's raise, model invests extra chips into a
# bigger pot. Some fraction of that extra is recovered through showdown
# equity. 0.50 ≈ heads-up coinflip; lower for bluff-heavy ranges, higher
# for value-heavy ranges. Used to scale the "premium cost" in the raise
# counterfactual.
_DEFAULT_CONTINUATION_EQUITY = 0.50

# ──────────────────────────────────────────────────────────────────────────────
# Strong-hand fold guard
# ──────────────────────────────────────────────────────────────────────────────
# Observation layout for 6p: Position(8) + PotOdds(3) + HandStrength(5) + Stack(2) + Equity(8)
# Equity block: [hand_strength, made_hand_cat, flush_draw, straight_draw, ...]
# ── Shared constants from DecisionEngine ─────────────────────────────────────
from src.decision_engine import (
    IDX_MADE_HAND as _MADE_HAND_IDX,
    IDX_EQUITY as _HAND_STRENGTH_IDX,
)

# Optional callback to perturb model_strategy after it's produced by strategy_fn.
# When non-None, scripts/research_weaknesses.py uses this to A/B-test strategy
# shifts on filtered decision buckets. Signature: (dp, base_strategy, legal) -> np.ndarray.
# Only fires on the postflop (non-preflop-range) branch of both
# analyze_decision_points and compute_showdown_ev.
_STRATEGY_PERTURBATION = None


# ──────────────────────────────────────────────────────────────────────────────
# Strategy functions
# ──────────────────────────────────────────────────────────────────────────────

def _fit_obs(deep_cfr, obs):
    """Truncate or zero-pad obs to match the model's expected observation size.

    Needed when the script's obs builder produces a different feature count
    than the model was trained with (e.g. older 50-dim model vs current 53-dim
    obs builder, or vice versa).
    """
    target = getattr(deep_cfr, 'observation_size', None)
    if target is None or target == len(obs):
        return obs
    if len(obs) > target:
        return obs[:target]
    padded = np.zeros(target, dtype=np.float32)
    padded[:len(obs)] = obs
    return padded


def _avg_strategy(deep_cfr, obs, player_idx, legal):
    return deep_cfr.get_strategy(_fit_obs(deep_cfr, obs), player_idx, legal)


def _current_strategy(deep_cfr, obs, player_idx, legal):
    obs = _fit_obs(deep_cfr, obs)
    advantages = deep_cfr.compute_counterfactual_values(obs, player_idx, legal)
    pos_adv = np.maximum(advantages, 0.0)
    s = pos_adv.sum()
    if s > 0:
        return pos_adv / s
    mask = np.zeros(deep_cfr.action_size, dtype=np.float32)
    mask[np.array(legal)] = 1.0
    return mask / mask.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Model loading (reused from validate_spots.py)
# ──────────────────────────────────────────────────────────────────────────────

def _make_engine(num_players: int, legacy_game_state: bool = False,
                  behavioral_stats: bool = False,
                  betting_line: bool = False,
                  threat_range_equity: bool = False) -> EnhancedPokerEngine:
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
        legacy_game_state=legacy_game_state,
        behavioral_stats=behavioral_stats,
        betting_line=betting_line,
        threat_range_equity=threat_range_equity,
    )


def _is_cql_checkpoint(ckpt: dict) -> bool:
    return ckpt.get('model_type') in ('cql', 'iql')


def _is_supervised_checkpoint(ckpt: dict) -> bool:
    return ('model_state_dict' in ckpt and 'advantage_networks' not in ckpt
            and not _is_cql_checkpoint(ckpt))


class _CQLWrapper:
    """Wraps a CQL QNetwork to look like DeepCFR for validation."""

    def __init__(self, q_net, obs_size, action_size, num_players,
                 iteration=0, temperature=1.0):
        self.action_size = action_size
        self.num_players = num_players
        self.obs_size = obs_size
        self.device = torch.device('cpu')
        self.iteration = iteration
        self.temperature = temperature
        self.q_net = q_net
        self.strategy_networks = [q_net] * num_players
        self.advantage_networks = [q_net] * num_players

    def get_strategy(self, observation, player_idx, legal_actions):
        # Pad observation if model expects more dims (e.g. equity_vs_odds)
        if len(observation) < self.obs_size:
            padded = np.zeros(self.obs_size, dtype=np.float32)
            padded[:len(observation)] = observation
            # Compute equity_vs_odds if we have the raw features
            if len(observation) >= 19 and self.obs_size > len(observation):
                pot_odds = observation[10]
                equity = observation[18]
                padded[len(observation)] = np.clip(equity - (1.0 - pot_odds), -1.0, 1.0)
            observation = padded
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(obs_t).numpy()[0]
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = q_values[a] / self.temperature
        masked -= masked.max()
        exp_s = np.exp(masked)
        return exp_s / exp_s.sum()


class _SupervisedWrapper:
    """Wraps a supervised StrategyNetwork to look like DeepCFR for validation."""

    def __init__(self, model, obs_size, action_size, num_players, iteration=0,
                 model_num_players=None):
        self.action_size = action_size
        self.num_players = num_players
        self.model_num_players = model_num_players or num_players
        self.device = torch.device('cpu')
        self.iteration = iteration
        self.strategy_networks = [model] * num_players
        self.advantage_networks = [model] * num_players
        self._pad_obs = None
        if self.model_num_players != num_players:
            from src.utils.obs_padding import pad_observation_to_6p
            self._pad_obs = pad_observation_to_6p

    def get_strategy(self, observation, player_idx, legal_actions):
        if self._pad_obs is not None:
            observation = self._pad_obs(observation, self.num_players)
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.strategy_networks[player_idx](obs_t).numpy()[0]
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = logits[a]
        masked -= masked.max()
        exp_s = np.exp(masked)
        return exp_s / exp_s.sum()


def _autodetect_players(model_path: Path, default: int = 6, force_legacy: bool = False):
    """Returns (num_players, legacy_game_state).

    For Deep CFR checkpoints, num_players is taken directly from the
    advantage-network count (one network per player position). Falls back
    to obs-size search for older / non-standard checkpoints.
    """
    try:
        ckpt = torch.load(str(model_path), map_location='cpu')
        if _is_cql_checkpoint(ckpt) or _is_supervised_checkpoint(ckpt):
            return ckpt.get('num_players', default), False
        # Deep CFR: trust network count
        adv = ckpt.get('advantage_networks')
        if adv and isinstance(adv, list) and len(adv) >= 2:
            return len(adv), force_legacy
        # Fallback: search by obs size
        input_size = ckpt['advantage_networks'][0]['network.0.weight'].shape[1]
        bh_opts = (False, True)
        bl_opts = (False, True)
        if force_legacy:
            for p in range(2, 10):
                for bh in bh_opts:
                    for bl in bl_opts:
                        if bl and not bh:
                            continue
                        if _make_engine(p, legacy_game_state=True, behavioral_stats=bh, betting_line=bl).get_observation_space_size() == input_size:
                            return p, True
        for p in range(2, 10):
            for bh in bh_opts:
                for bl in bl_opts:
                    if bl and not bh:
                        continue
                    if _make_engine(p, behavioral_stats=bh, betting_line=bl).get_observation_space_size() == input_size:
                        return p, False
        for p in range(2, 10):
            for bh in bh_opts:
                for bl in bl_opts:
                    if bl and not bh:
                        continue
                    if _make_engine(p, legacy_game_state=True, behavioral_stats=bh, betting_line=bl).get_observation_space_size() == input_size:
                        return p, True
    except Exception:
        pass
    return default, False


def _load_model(model_path: Path, config: dict, num_players: int,
                legacy: bool = False, model_num_players: int = None,
                behavioral_stats: bool = False,
                betting_line: bool = False,
                threat_range_equity: bool = False):
    """Load model. If model_num_players is set and differs from num_players,
    observations will be padded from num_players to model_num_players format."""
    ckpt = torch.load(str(model_path), map_location='cpu')
    mnp = model_num_players  # the player count the model was trained on
    if _is_cql_checkpoint(ckpt):
        from train_cql import QNetwork
        obs_size = ckpt['obs_size']
        action_size = ckpt['action_size']
        hidden = ckpt.get('hidden_layers', config['network']['hidden_layers'])
        dropout = ckpt.get('dropout', 0.1)
        q_net = QNetwork(obs_size, action_size, hidden, dropout)
        q_net.load_state_dict(ckpt['model_state_dict'])
        q_net.eval()
        temperature = ckpt.get('temperature', 1.0)
        return _CQLWrapper(q_net, obs_size, action_size, num_players,
                           ckpt.get('epoch', 0), temperature)
    if _is_supervised_checkpoint(ckpt):
        # Use model's trained player count for obs_size/action_size
        model_np = mnp or ckpt.get('num_players', num_players)
        engine_for_model = _make_engine(model_np, legacy_game_state=legacy,
                                         behavioral_stats=behavioral_stats,
                                         betting_line=betting_line,
                                         threat_range_equity=threat_range_equity)
        obs_size = ckpt.get('obs_size', engine_for_model.get_observation_space_size())
        action_size = ckpt.get('action_size', engine_for_model.get_action_space_size())
        hidden = ckpt.get('hidden_layers', config['network']['hidden_layers'])
        dropout = ckpt.get('dropout', 0.1)
        model = StrategyNetwork(obs_size, action_size, hidden, dropout)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        return _SupervisedWrapper(
            model, obs_size, action_size, num_players,
            ckpt.get('epoch', 0),
            model_num_players=model_np if model_np != num_players else None)
    engine = _make_engine(num_players, legacy_game_state=legacy,
                          behavioral_stats=behavioral_stats,
                          betting_line=betting_line,
                          threat_range_equity=threat_range_equity)
    # Auto-detect hidden layer dims from checkpoint state_dict (layers 0/5/9
    # are the three Linear layers between residual blocks). Falls back to
    # config-file value for older / non-standard architectures.
    hidden = config['network']['hidden_layers']
    try:
        adv = ckpt['advantage_networks'][0]
        hidden = [
            adv['network.0.weight'].shape[0],
            adv['network.5.weight'].shape[0],
            adv['network.9.weight'].shape[0],
        ]
    except (KeyError, IndexError):
        pass
    deep_cfr = DeepCFR(
        observation_size=engine.get_observation_space_size(),
        action_size=engine.get_action_space_size(),
        num_players=num_players,
        hidden_layers=hidden,
        device='cpu',
    )
    deep_cfr.load_models(str(model_path))
    # HH parser uses 5 actions (0-4 incl ALL_IN) but fast engine may only
    # have 4.  Expand action_size so strategy arrays can hold all HH actions.
    deep_cfr.action_size = max(5, deep_cfr.action_size)
    return deep_cfr


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_decision_points(deep_cfr, decision_points: list,
                            strategy_fn, num_players: int,
                            preflop_ranges=None,
                            good_players=None, bad_players=None):
    """Analyze model strategy at each decision point from hand histories.

    Returns dicts of metrics for reporting.
    """
    good_players = good_players or set()
    bad_players = bad_players or set()

    # Per-street action agreement
    agreement = defaultdict(lambda: {'match': 0, 'total': 0})
    # Split by player quality: 'good', 'bad', 'unknown'
    agreement_by_quality = {
        'good': defaultdict(lambda: {'match': 0, 'total': 0}),
        'bad': defaultdict(lambda: {'match': 0, 'total': 0}),
        'unknown': defaultdict(lambda: {'match': 0, 'total': 0}),
    }
    # Pot-weighted agreement
    pot_weighted = {'weighted_match': 0.0, 'weighted_total': 0.0}
    pot_weighted_by_quality = {
        'good': {'weighted_match': 0.0, 'weighted_total': 0.0},
        'bad': {'weighted_match': 0.0, 'weighted_total': 0.0},
    }
    # Model probability assigned to actual action
    p_actual_list = []
    # Spot-based samples: cat -> [(model_strategy, legal_set, ctx)]
    model_samples = defaultdict(list)
    # NL5 population samples: cat -> [(action_onehot, legal_set, ctx)]
    nl5_samples = defaultdict(list)
    # Position stats for model
    model_pos = defaultdict(lambda: {
        'hands': 0, 'vpip': 0.0, 'pfr': 0.0,
        'three_bet': 0.0, 'three_bet_opp': 0,
        'open_raise': 0.0, 'open_opp': 0,
    })
    # Position stats for NL5
    nl5_pos = defaultdict(lambda: {
        'hands': 0, 'vpip': 0.0, 'pfr': 0.0,
        'three_bet': 0.0, 'three_bet_opp': 0,
        'open_raise': 0.0, 'open_opp': 0,
    })
    # Track which (hand, player) combos we've counted for position stats
    seen_preflop = set()

    # Position label → relative position for preflop ranges
    _POS_TO_REL = {'BTN': 0, 'SB': 1, 'BB': 2, 'UTG': 3, 'MP': 4, 'CO': 5}

    for dp in decision_points:
        legal = dp.legal_actions
        if not legal:
            continue

        # Use relative position as player_idx for model query
        rel_pos = _pos_to_idx(dp.position_label, num_players)

        # Preflop range override
        if (dp.street == 'preflop' and preflop_ranges is not None
                and dp.hole_cards and len(dp.hole_cards) >= 2):
            pf_rel = _POS_TO_REL.get(dp.position_label, 0)
            pot_bb = dp.pot_dollars / 0.05 if dp.pot_dollars > 0 else 1.5
            model_strategy = preflop_ranges.get_strategy(
                dp.hole_cards, pf_rel, pot_bb, legal, deep_cfr.action_size)
        else:
            model_strategy = strategy_fn(dp.observation, rel_pos, legal)
            if _STRATEGY_PERTURBATION is not None:
                model_strategy = _STRATEGY_PERTURBATION(dp, model_strategy, legal)

        actual = dp.actual_action
        street = dp.street

        # Action agreement
        model_top = int(np.argmax(model_strategy))
        agreement[street]['total'] += 1
        agreement['overall']['total'] += 1
        matched = (model_top == actual)
        if matched:
            agreement[street]['match'] += 1
            agreement['overall']['match'] += 1

        # Player quality split
        pname = dp.player_name
        if pname in good_players:
            quality = 'good'
        elif pname in bad_players:
            quality = 'bad'
        else:
            quality = 'unknown'
        agreement_by_quality[quality]['overall']['total'] += 1
        agreement_by_quality[quality][street]['total'] += 1
        if matched:
            agreement_by_quality[quality]['overall']['match'] += 1
            agreement_by_quality[quality][street]['match'] += 1

        # Pot-weighted agreement (bigger pots matter more)
        pot_bb = dp.pot_dollars / 0.05 if dp.pot_dollars > 0 else 1.0
        pot_weighted['weighted_total'] += pot_bb
        if matched:
            pot_weighted['weighted_match'] += pot_bb
        if quality in ('good', 'bad'):
            pot_weighted_by_quality[quality]['weighted_total'] += pot_bb
            if matched:
                pot_weighted_by_quality[quality]['weighted_match'] += pot_bb

        # P(actual)
        if actual < len(model_strategy):
            p_actual_list.append(float(model_strategy[actual]))

        # Classify spot
        if dp.hole_cards:
            if street == 'preflop':
                cat = classify_preflop(dp.hole_cards)
            else:
                cat = classify_postflop(dp.hole_cards, dp.community_cards)
        else:
            cat = None

        # Build context for spot stats
        max_bet_chips = 0
        if FOLD in legal:
            # If fold is legal, there's a bet to face
            max_bet_chips = 4  # approximate
        ctx = {
            'pos_label': dp.position_label,
            'facing_preflop_raise': street == 'preflop' and FOLD in legal,
            'facing_postflop_bet': street != 'preflop' and FOLD in legal,
            'call_amount': 1 if FOLD in legal else 0,
        }

        legal_set = set(legal)

        if cat is not None:
            model_samples[cat].append((model_strategy.copy(), legal_set, ctx))

            # NL5 actual action as one-hot strategy
            nl5_strat = np.zeros(deep_cfr.action_size, dtype=np.float32)
            nl5_strat[actual] = 1.0
            nl5_samples[cat].append((nl5_strat, legal_set, ctx))

        # Position stats (preflop only, first action per player per hand)
        dp_key = (dp.hand_id, dp.player_name)
        if street == 'preflop' and dp_key not in seen_preflop:
            seen_preflop.add(dp_key)
            pos = dp.position_label

            # Model position stats
            model_pos[pos]['hands'] += 1
            n_act = len(model_strategy)
            fold_p = float(model_strategy[FOLD]) if FOLD in legal_set and FOLD < n_act else 0.0
            raise_p = sum(float(model_strategy[a]) for a in AGGRESSIVE if a in legal_set and a < n_act)

            is_bb_free = (pos == 'BB' and FOLD not in legal_set)
            if not is_bb_free:
                model_pos[pos]['vpip'] += (1.0 - fold_p)
            model_pos[pos]['pfr'] += raise_p

            facing = ctx.get('facing_preflop_raise', False)
            if facing:
                model_pos[pos]['three_bet_opp'] += 1
                model_pos[pos]['three_bet'] += raise_p
            if not facing and pos != 'BB':
                model_pos[pos]['open_opp'] += 1
                model_pos[pos]['open_raise'] += raise_p

            # NL5 position stats
            nl5_pos[pos]['hands'] += 1
            actual_is_fold = (actual == FOLD)
            actual_is_raise = (actual in AGGRESSIVE)

            if not is_bb_free:
                nl5_pos[pos]['vpip'] += (0.0 if actual_is_fold else 1.0)
            nl5_pos[pos]['pfr'] += (1.0 if actual_is_raise else 0.0)

            if facing:
                nl5_pos[pos]['three_bet_opp'] += 1
                nl5_pos[pos]['three_bet'] += (1.0 if actual_is_raise else 0.0)
            if not facing and pos != 'BB':
                nl5_pos[pos]['open_opp'] += 1
                nl5_pos[pos]['open_raise'] += (1.0 if actual_is_raise else 0.0)

    return {
        'agreement': dict(agreement),
        'agreement_by_quality': {k: dict(v) for k, v in agreement_by_quality.items()},
        'pot_weighted': pot_weighted,
        'pot_weighted_by_quality': pot_weighted_by_quality,
        'p_actual': p_actual_list,
        'model_samples': dict(model_samples),
        'nl5_samples': dict(nl5_samples),
        'model_pos': dict(model_pos),
        'nl5_pos': dict(nl5_pos),
    }


def _pos_to_idx(pos_label: str, num_players: int) -> int:
    """Convert position label to player_idx for model query."""
    mapping = {'BTN': 0, 'SB': 1, 'BB': 2, 'UTG': 3, 'MP': 4, 'CO': 5}
    return mapping.get(pos_label, 0) % num_players


# ──────────────────────────────────────────────────────────────────────────────
# NL5 population stats from ALL hands (not just showdown)
# ──────────────────────────────────────────────────────────────────────────────

_POS_LABELS_6 = {0: 'BTN', 1: 'SB', 2: 'BB', 3: 'UTG', 4: 'MP', 5: 'CO'}

def compute_nl5_population_stats(all_6p_hands: list, num_players: int = 6):
    """Compute NL5 population action frequencies from ALL 6-player hands.

    This uses raw parsed actions — no observation building or cards needed.
    Gives unbiased fold/raise rates for the population.
    """
    from src.hand_history.action_mapper import FOLD as _FOLD
    from src.hand_history.models import dollars_to_chips

    nl5_pos = defaultdict(lambda: {
        'hands': 0, 'vpip': 0.0, 'pfr': 0.0,
        'three_bet': 0.0, 'three_bet_opp': 0,
        'open_raise': 0.0, 'open_opp': 0,
    })

    # Per-street action counts: action_type -> count
    action_counts = defaultdict(lambda: defaultdict(int))

    for hand in all_6p_hands:
        active = hand.active_players
        if len(active) != num_players:
            continue

        # Find button
        btn_idx = 0
        for i, p in enumerate(active):
            if p.seat == hand.button_seat:
                btn_idx = i
                break

        # Track preflop: who acted, was there a raise before
        seen_preflop = set()
        preflop_raised = False  # was there a raise before this action?

        for action in hand.actions:
            if action.is_blind:
                continue

            street = action.street

            # Find player index
            pidx = None
            for i, p in enumerate(active):
                if p.name == action.player_name:
                    pidx = i
                    break
            if pidx is None:
                continue

            rel_pos = (pidx - btn_idx) % num_players
            pos = _POS_LABELS_6.get(rel_pos, 'MP')

            # Count actions per street
            action_counts[street][action.action_type] += 1

            # Preflop position stats (first action per player)
            if street == 'preflop' and action.player_name not in seen_preflop:
                seen_preflop.add(action.player_name)
                nl5_pos[pos]['hands'] += 1

                is_fold = action.action_type == 'fold'
                is_raise = action.action_type in ('raise', 'bet') or action.is_all_in
                is_bb_free = (pos == 'BB' and action.action_type == 'check')

                if not is_bb_free:
                    nl5_pos[pos]['vpip'] += (0.0 if is_fold else 1.0)
                nl5_pos[pos]['pfr'] += (1.0 if is_raise else 0.0)

                if preflop_raised:
                    nl5_pos[pos]['three_bet_opp'] += 1
                    nl5_pos[pos]['three_bet'] += (1.0 if is_raise else 0.0)
                elif pos != 'BB':
                    nl5_pos[pos]['open_opp'] += 1
                    nl5_pos[pos]['open_raise'] += (1.0 if is_raise else 0.0)

                if is_raise:
                    preflop_raised = True

    return dict(nl5_pos), dict(action_counts)


# ──────────────────────────────────────────────────────────────────────────────
# Report printing
# ──────────────────────────────────────────────────────────────────────────────

def compute_showdown_ev(showdown_hands: list, all_decision_points: list,
                        strategy_fn, deep_cfr, num_players: int,
                        preflop_ranges=None,
                        good_players=None, bad_players=None,
                        p_fold_to_raise: float = _DEFAULT_P_FOLD,
                        continuation_equity: float = _DEFAULT_CONTINUATION_EQUITY):
    """Estimate model's EV advantage vs NL5 players at showdown hands.

    For each showdown player with known hole cards we walk their decision
    sequence and stop at the first deviation from what the player did:

    - Model FOLD where player did not → model_net = -(invested up to fold).
      Avoided loss / missed gain accounted by player_net sign.
    - Model RAISE (½/pot/all-in) where player called/checked → assumed crude
      fold-equity counterfactual: with prob p_fold_to_raise opponent folds and
      model collects pot_at_action; with prob (1 - p_fold_to_raise) the hand
      plays out as the player's path but with extra chips invested due to
      the raise size. Net contribution is added to the player's actual net.
    - Otherwise → model_net = player_net (same as player).

    The raise counterfactual is approximate (no real opponent simulation),
    but it gives aggressive models partial credit for fold equity that the
    fold-only baseline ignored.

    Returns dict with EV stats including raise-deviation breakdowns.
    """
    good_players = good_players or set()
    bad_players = bad_players or set()

    # Group decision points by (hand_id, player_name), preserving order
    dp_by_hand_player = defaultdict(list)
    for dp in all_decision_points:
        dp_by_hand_player[(dp.hand_id, dp.player_name)].append(dp)

    # Per-hand results
    player_ev_diffs = []
    avoided_losses = []
    missed_gains = []
    same_play = []
    model_nets = []
    player_nets = []

    # Calling efficiency / fold accuracy
    continue_wins = 0
    continue_losses = 0
    fold_correct = 0
    fold_wrong = 0

    # Raise-upgrade deviation tracking (model raises where player didn't)
    raise_upgrade_count = 0
    raise_fe_gains = []   # estimated fold-equity gains
    raise_premium_costs = []  # estimated extra investment when not folded
    street_raises = defaultdict(lambda: {
        'count': 0, 'fe_gain': 0.0, 'premium_cost': 0.0,
    })

    ev_by_quality = {
        'good': {'ev_diffs': [], 'model_nets': [], 'player_nets': []},
        'bad': {'ev_diffs': [], 'model_nets': [], 'player_nets': []},
        'unknown': {'ev_diffs': [], 'model_nets': [], 'player_nets': []},
    }

    street_folds = defaultdict(lambda: {'count': 0, 'saved': 0.0, 'missed': 0.0})

    _POS_TO_REL_EV = {'BTN': 0, 'SB': 1, 'BB': 2,
                      'UTG': 3, 'MP': 4, 'CO': 5}

    for hand in showdown_hands:
        invested = _compute_all_invested(hand)

        collected = defaultdict(float)
        for w in hand.result.winners:
            collected[w['name']] += w['amount_dollars']

        # Cache invested-timeline once per (hand, player) — avoids O(n²) recompute
        invested_timelines = {}

        for player_name in hand.shown_hands:
            player_invested = invested.get(player_name, 0.0)
            player_collected = collected.get(player_name, 0.0)
            player_net = player_collected - player_invested

            dps = dp_by_hand_player.get((hand.hand_id, player_name), [])
            if not dps:
                continue

            timeline = invested_timelines.get(player_name)
            if timeline is None:
                timeline = _build_invested_timeline(hand, player_name)
                invested_timelines[player_name] = timeline

            # Walk decisions: stop at first deviation (fold or raise upgrade)
            model_would_fold = False
            invested_at_fold = 0.0
            fold_street = None

            raise_upgrade = False
            raise_pot_at_action = 0.0
            raise_action_code = None
            raise_street = None

            for dp in dps:
                legal = dp.legal_actions
                if not legal:
                    continue

                running_invested = timeline.get(dp.action_index, 0.0)
                rel_pos = _pos_to_idx(dp.position_label, num_players)

                if (dp.street == 'preflop' and preflop_ranges is not None
                        and dp.hole_cards and len(dp.hole_cards) >= 2):
                    pf_rel = _POS_TO_REL_EV.get(dp.position_label, 0)
                    pot_bb = (dp.pot_dollars / _BB_DOLLARS
                              if dp.pot_dollars > 0 else 1.5)
                    model_strategy = preflop_ranges.get_strategy(
                        dp.hole_cards, pf_rel, pot_bb, legal,
                        deep_cfr.action_size)
                else:
                    model_strategy = strategy_fn(dp.observation, rel_pos, legal)
                    if _STRATEGY_PERTURBATION is not None:
                        model_strategy = _STRATEGY_PERTURBATION(dp, model_strategy, legal)

                model_top = int(np.argmax(model_strategy))
                actual = dp.actual_action

                if model_top == FOLD and FOLD in legal:
                    model_would_fold = True
                    invested_at_fold = running_invested
                    fold_street = dp.street
                    break

                # Raise upgrade: model raises where player did not raise
                # (and didn't fold either — folds are handled above).
                if (model_top in AGGRESSIVE
                        and actual not in AGGRESSIVE
                        and actual != FOLD):
                    raise_upgrade = True
                    raise_pot_at_action = dp.pot_dollars
                    raise_action_code = model_top
                    raise_street = dp.street
                    break

            if model_would_fold:
                model_net = -invested_at_fold
            elif raise_upgrade:
                # Crude fold-equity counterfactual:
                #   p_fold * pot_won_immediately            (gain when opp folds)
                # - (1 - p_fold) * extra_invest * (1 - cont_equity)
                #                                            (cost when called)
                # cont_equity = expected pot-share recovered through showdown
                # in the bigger called pot. 0.50 ≈ neutral assumption.
                fe_gain = p_fold_to_raise * raise_pot_at_action
                size_mult = _RAISE_SIZE_MULT.get(raise_action_code, 0.5)
                extra_invest = raise_pot_at_action * size_mult
                premium = ((1.0 - p_fold_to_raise) * extra_invest
                           * (1.0 - continuation_equity))
                model_net = player_net + fe_gain - premium

                raise_upgrade_count += 1
                raise_fe_gains.append(fe_gain)
                raise_premium_costs.append(premium)
                street_raises[raise_street]['count'] += 1
                street_raises[raise_street]['fe_gain'] += fe_gain
                street_raises[raise_street]['premium_cost'] += premium
            else:
                model_net = player_net

            ev_diff = model_net - player_net
            player_ev_diffs.append(ev_diff)
            model_nets.append(model_net)
            player_nets.append(player_net)

            if player_name in good_players:
                q = 'good'
            elif player_name in bad_players:
                q = 'bad'
            else:
                q = 'unknown'
            ev_by_quality[q]['ev_diffs'].append(ev_diff)
            ev_by_quality[q]['model_nets'].append(model_net)
            ev_by_quality[q]['player_nets'].append(player_net)

            if model_would_fold:
                if player_net < 0:
                    saved = (-player_net) - invested_at_fold
                    avoided_losses.append(saved)
                    street_folds[fold_street]['count'] += 1
                    street_folds[fold_street]['saved'] += saved
                    fold_correct += 1
                else:
                    missed = player_net + invested_at_fold
                    missed_gains.append(missed)
                    street_folds[fold_street]['count'] += 1
                    street_folds[fold_street]['missed'] += missed
                    fold_wrong += 1
            else:
                same_play.append(player_net)
                if player_net >= 0:
                    continue_wins += 1
                else:
                    continue_losses += 1

    total = len(player_ev_diffs)
    return {
        'total_showdown_players': total,
        'ev_diffs': player_ev_diffs,
        'model_nets': model_nets,
        'player_nets': player_nets,
        'avoided_losses': avoided_losses,
        'missed_gains': missed_gains,
        'same_play': same_play,
        'street_folds': dict(street_folds),
        'continue_wins': continue_wins,
        'continue_losses': continue_losses,
        'fold_correct': fold_correct,
        'fold_wrong': fold_wrong,
        'ev_by_quality': ev_by_quality,
        # Raise-upgrade counterfactual
        'raise_upgrade_count': raise_upgrade_count,
        'raise_fe_gains': raise_fe_gains,
        'raise_premium_costs': raise_premium_costs,
        'street_raises': dict(street_raises),
        'p_fold_used': p_fold_to_raise,
    }


def _compute_all_invested(hand) -> dict:
    """Compute total $ invested per player from hand actions."""
    # Track per-player per-street running bet to handle raise "to" amounts
    invested = defaultdict(float)
    street_bet = defaultdict(float)  # (player, street) -> current bet this street
    current_street = 'preflop'

    for action in hand.actions:
        name = action.player_name
        street = action.street

        # Street changed — reset per-street bets
        if street != current_street:
            street_bet.clear()
            current_street = street

        if action.action_type in ('blind_sb', 'blind_bb'):
            invested[name] += action.amount_dollars
            street_bet[name] = action.amount_dollars
        elif action.action_type == 'call':
            invested[name] += action.amount_dollars
            street_bet[name] += action.amount_dollars
        elif action.action_type == 'bet':
            invested[name] += action.amount_dollars
            street_bet[name] = action.amount_dollars
        elif action.action_type == 'raise':
            # amount_dollars is "to" amount (total bet this street)
            prev_bet = street_bet.get(name, 0.0)
            increment = action.amount_dollars - prev_bet
            invested[name] += increment
            street_bet[name] = action.amount_dollars

    return dict(invested)


def _invested_up_to_action(hand, player_name: str, action_index: int) -> float:
    """Compute $ invested by player up to (but not including) action at action_index.

    Kept for backwards compatibility. Per-decision callers should prefer
    _build_invested_timeline() — O(n) once per (hand, player) instead of O(n²).
    """
    return _build_invested_timeline(hand, player_name).get(action_index, 0.0)


def _build_invested_timeline(hand, player_name: str) -> dict:
    """Return {action_index: invested_up_to_but_not_including_that_action}.

    One pass over hand.actions for the named player. The returned dict can
    be reused across all of that player's decision points in the hand.
    """
    timeline = {}
    invested = 0.0
    street_bet = 0.0
    current_street = 'preflop'
    idx = 0

    for action in hand.actions:
        if action.street != current_street:
            street_bet = 0.0
            current_street = action.street

        if action.player_name == player_name:
            timeline[idx] = invested
            if action.action_type in ('blind_sb', 'blind_bb'):
                invested += action.amount_dollars
                street_bet = action.amount_dollars
            elif action.action_type == 'call':
                invested += action.amount_dollars
                street_bet += action.amount_dollars
            elif action.action_type == 'bet':
                invested += action.amount_dollars
                street_bet = action.amount_dollars
            elif action.action_type == 'raise':
                increment = action.amount_dollars - street_bet
                invested += increment
                street_bet = action.amount_dollars
            idx += 1

    return timeline


def print_report(metrics: dict, parse_stats: dict, deep_cfr, num_dp: int):
    W = 80
    print()
    print('=' * W)
    print(f'  HAND HISTORY VALIDATION  (model iter {deep_cfr.iteration})')
    print('=' * W)

    # ── Parse stats ───────────────────────────────────────────────────────
    print(f"\n  Files parsed:          {parse_stats['files']}")
    print(f"  Total hands:           {parse_stats['total_hands']}")
    all_6p = parse_stats.get('all_6p_hands', parse_stats['filtered_hands'])
    print(f"  6-player hands:        {all_6p}  (population stats source)")
    print(f"  6-player showdown:     {parse_stats['filtered_hands']}"
          f"  ({parse_stats['filtered_hands']/max(parse_stats['total_hands'],1)*100:.1f}%)"
          f"  (model comparison source)")
    print(f"  Decision points:       {num_dp}")

    # ── Action Agreement ──────────────────────────────────────────────────
    agreement = metrics['agreement']
    print(f'\n  ── ACTION AGREEMENT (model top action == player action) ──')
    for key in ['overall', 'preflop', 'flop', 'turn', 'river']:
        if key in agreement and agreement[key]['total'] > 0:
            a = agreement[key]
            rate = a['match'] / a['total']
            label = key.capitalize() if key != 'overall' else 'Overall'
            print(f"  {label:<12} {rate:>6.1%}   ({a['match']}/{a['total']})")

    # Pot-weighted agreement
    pw = metrics.get('pot_weighted', {})
    if pw.get('weighted_total', 0) > 0:
        pw_rate = pw['weighted_match'] / pw['weighted_total']
        print(f"  Pot-weighted {pw_rate:>6.1%}   (bigger pots count more)")

    # ── Agreement by player quality ──────────────────────────────────────
    abq = metrics.get('agreement_by_quality', {})
    has_quality = any(
        abq.get(q, {}).get('overall', {}).get('total', 0) > 0
        for q in ('good', 'bad')
    )
    if has_quality:
        print(f"\n  ── AGREEMENT BY PLAYER QUALITY ──")
        print(f"  {'Quality':<10} {'Overall':>8}  {'Preflop':>8}  {'Flop':>8}  "
              f"{'Turn':>8}  {'River':>8}  {'PotWtd':>8}")
        print('  ' + '-' * 68)
        pwq = metrics.get('pot_weighted_by_quality', {})
        for q_name in ['good', 'bad', 'unknown']:
            qa = abq.get(q_name, {})
            ov = qa.get('overall', {})
            if ov.get('total', 0) == 0:
                continue
            parts = [f"  {q_name.capitalize():<10}"]
            for key in ['overall', 'preflop', 'flop', 'turn', 'river']:
                a = qa.get(key, {})
                if a.get('total', 0) > 0:
                    rate = a['match'] / a['total']
                    parts.append(f"{rate:>7.1%}")
                else:
                    parts.append(f"{'—':>8}")
            # Pot-weighted for this quality
            pq = pwq.get(q_name, {})
            if pq.get('weighted_total', 0) > 0:
                pwr = pq['weighted_match'] / pq['weighted_total']
                parts.append(f"{pwr:>7.1%}")
            else:
                parts.append(f"{'—':>8}")
            print('  '.join(parts))
            n = ov['total']
            print(f"  {'':>10} ({n} decisions)")

    # ── Model confidence in actual action ─────────────────────────────────
    p_actual = metrics['p_actual']
    if p_actual:
        avg_p = np.mean(p_actual)
        med_p = np.median(p_actual)
        print(f"\n  ── MODEL CONFIDENCE IN ACTUAL ACTION ──")
        print(f"  Avg P(actual):    {avg_p:.3f}")
        print(f"  Median P(actual): {med_p:.3f}")

    # ── Per-spot strategy comparison ──────────────────────────────────────
    print(f"\n  ── PER-SPOT STRATEGY: MODEL vs NL5 (showdown players only) ──")
    print(f"  Note: NL5 stats are from showdown players — fold% is biased low")
    hdr = (f"  {'Spot':<30} {'N':>5}  "
           f"{'Mfold%':>7} {'Pfold%':>7}  "
           f"{'Maggr%':>7} {'Paggr%':>7}  GTO")
    sep = '  ' + '-' * (W - 4)

    current_street = None
    model_samples = metrics['model_samples']
    nl5_samples = metrics['nl5_samples']

    # All SPOTS are postflop categories
    print(f"\n  ── POST-FLOP ──")
    print(hdr)
    print(sep)

    for spot in SPOTS:
        cat = spot['category']
        label = spot['label'][:30]

        m_stats = _compute_spot_stats(model_samples.get(cat, []))
        n_stats = _compute_spot_stats(nl5_samples.get(cat, []))

        if m_stats is None or m_stats['n'] == 0:
            print(f"  {label:<30}     —   (no samples)")
            continue

        n = m_stats['n']
        m_fold = m_stats['fold_pct']
        m_aggr = m_stats['aggr_pct']
        n_fold = n_stats['fold_pct'] if n_stats else 0
        n_aggr = n_stats['aggr_pct'] if n_stats else 0

        gto_ok = _spot_passes(spot, m_fold, m_aggr)
        gto_str = 'PASS' if gto_ok else 'FAIL'
        if n < 10:
            gto_str += '?'

        print(f"  {label:<30} {n:>5}  "
              f"{m_fold:>6.1%}  {n_fold:>6.1%}  "
              f"{m_aggr:>6.1%}  {n_aggr:>6.1%}  {gto_str}")

    # ── Position stats ────────────────────────────────────────────────────
    model_pos = metrics['model_pos']
    nl5_pos = metrics['nl5_pos']

    if model_pos:
        print(f"\n  ── POSITION STATS: MODEL vs NL5 (all {parse_stats.get('all_6p_hands','?')} hands) ──")
        pos_hdr = (f"  {'Pos':<5}  "
                   f"{'M-VPIP':>7} {'P-VPIP':>7}  "
                   f"{'M-PFR':>6} {'P-PFR':>6}  "
                   f"{'M-3bet':>7} {'P-3bet':>7}  GTO-VPIP")
        print(pos_hdr)
        print('  ' + '-' * (W - 4))

        for pos in ['UTG', 'MP', 'CO', 'BTN', 'SB', 'BB']:
            mp = model_pos.get(pos)
            np_ = nl5_pos.get(pos)
            if not mp or mp['hands'] < 5:
                continue

            h = mp['hands']
            m_vpip = mp['vpip'] / h
            m_pfr = mp['pfr'] / h
            m_3bet = (mp['three_bet'] / mp['three_bet_opp']
                      if mp['three_bet_opp'] > 0 else float('nan'))

            nh = np_['hands'] if np_ else 1
            n_vpip = np_['vpip'] / nh if np_ else 0
            n_pfr = np_['pfr'] / nh if np_ else 0
            n_3bet = (np_['three_bet'] / np_['three_bet_opp']
                      if np_ and np_['three_bet_opp'] > 0 else float('nan'))

            gto = GTO_VPIP.get(pos, float('nan'))
            gto_str = f'{gto:>7.0%}' if not np.isnan(gto) else '    n/a'

            m3s = f'{m_3bet:>6.1%}' if not np.isnan(m_3bet) else '   n/a'
            n3s = f'{n_3bet:>6.1%}' if not np.isnan(n_3bet) else '   n/a'

            print(f"  {pos:<5}  "
                  f"{m_vpip:>6.1%}  {n_vpip:>6.1%}  "
                  f"{m_pfr:>6.1%}  {n_pfr:>6.1%}  "
                  f"{m3s:>7}  {n3s:>7}  {gto_str}")

    # ── Exploitability signals ────────────────────────────────────────────
    print(f"\n  ── EXPLOITABILITY SIGNALS (largest model vs NL5 divergence) ──")
    divergences = []
    for spot in SPOTS:
        cat = spot['category']
        m_stats = _compute_spot_stats(model_samples.get(cat, []))
        n_stats = _compute_spot_stats(nl5_samples.get(cat, []))
        if m_stats and n_stats and m_stats['n'] >= 10:
            fold_diff = m_stats['fold_pct'] - n_stats['fold_pct']
            aggr_diff = m_stats['aggr_pct'] - n_stats['aggr_pct']
            total_diff = abs(fold_diff) + abs(aggr_diff)
            direction = []
            if abs(fold_diff) > 0.05:
                direction.append(f"fold {'higher' if fold_diff > 0 else 'lower'} by {abs(fold_diff):.0%}")
            if abs(aggr_diff) > 0.05:
                direction.append(f"aggr {'higher' if aggr_diff > 0 else 'lower'} by {abs(aggr_diff):.0%}")
            if direction:
                gto_ok = _spot_passes(spot, m_stats['fold_pct'], m_stats['aggr_pct'])
                quality = 'GTO-aligned' if gto_ok else 'exploitable'
                divergences.append((total_diff, spot['label'][:35],
                                    ', '.join(direction), quality))

    divergences.sort(reverse=True)
    for i, (_, label, desc, quality) in enumerate(divergences[:8]):
        print(f"  {i+1}. {label}: model {desc} ({quality})")

    if not divergences:
        print("  (not enough data for divergence analysis)")

    # ── Showdown EV Analysis ─────────────────────────────────────────────
    ev_data = metrics.get('showdown_ev')
    if ev_data and ev_data['total_showdown_players'] > 0:
        print(f"\n  ── SHOWDOWN EV ANALYSIS ──")
        total_sp = ev_data['total_showdown_players']
        model_total = sum(ev_data['model_nets'])
        player_total = sum(ev_data['player_nets'])
        ev_diff_total = sum(ev_data['ev_diffs'])

        print(f"  Showdown player-hands analyzed:  {total_sp}")
        print(f"  NL5 total net:        ${player_total:>+8.2f}")
        print(f"  Model estimated net:  ${model_total:>+8.2f}")
        print(f"  EV difference:        ${ev_diff_total:>+8.2f}"
              f"  ({ev_diff_total/total_sp:>+.4f} $/hand)")

        avoided = ev_data['avoided_losses']
        missed = ev_data['missed_gains']
        same = ev_data['same_play']

        n_fold = len(avoided) + len(missed)
        print(f"\n  Model folds where player continued:  {n_fold}"
              f"  ({n_fold/max(total_sp,1)*100:.1f}%)")
        if avoided:
            print(f"    Avoided losses (player lost, model folds): {len(avoided)}"
                  f"  total saved: ${sum(avoided):>+.2f}")
        if missed:
            print(f"    Missed gains  (player won, model folds):   {len(missed)}"
                  f"  total missed: ${-sum(missed):>+.2f}")
        print(f"  Model plays same as player: {len(same)}"
              f"  ({len(same)/max(total_sp,1)*100:.1f}%)")

        # Per-street fold breakdown
        sf = ev_data['street_folds']
        if sf:
            print(f"\n  Fold-by-street breakdown:")
            for street in ['preflop', 'flop', 'turn', 'river']:
                if street in sf:
                    s = sf[street]
                    net = s['saved'] - s['missed']
                    print(f"    {street:<8} folds: {s['count']:>4}"
                          f"   saved: ${s['saved']:>7.2f}"
                          f"   missed: ${s['missed']:>7.2f}"
                          f"   net: ${net:>+.2f}")

        # ── Raise-upgrade counterfactual ──────────────────────────────
        ru_count = ev_data.get('raise_upgrade_count', 0)
        if ru_count > 0:
            fe_total = sum(ev_data['raise_fe_gains'])
            prem_total = sum(ev_data['raise_premium_costs'])
            net_raise = fe_total - prem_total
            p_fold = ev_data.get('p_fold_used', _DEFAULT_P_FOLD)
            print(f"\n  Model raises where player called/checked:  {ru_count}"
                  f"  ({ru_count/max(total_sp,1)*100:.1f}%)")
            print(f"    Assumed fold equity p_fold = {p_fold:.2f}")
            print(f"    Estimated FE gain:    ${fe_total:>+8.2f}")
            print(f"    Estimated premium:    ${prem_total:>+8.2f}")
            print(f"    Net raise estimate:   ${net_raise:>+8.2f}")

            sr = ev_data.get('street_raises', {})
            if sr:
                print(f"\n  Raise-by-street breakdown:")
                for street in ['preflop', 'flop', 'turn', 'river']:
                    if street in sr:
                        s = sr[street]
                        net = s['fe_gain'] - s['premium_cost']
                        print(f"    {street:<8} raises: {s['count']:>4}"
                              f"   FE: ${s['fe_gain']:>7.2f}"
                              f"   prem: ${s['premium_cost']:>7.2f}"
                              f"   net: ${net:>+.2f}")

        # ── Fold accuracy & calling efficiency ────────────────────────
        fc = ev_data.get('fold_correct', 0)
        fw = ev_data.get('fold_wrong', 0)
        cw = ev_data.get('continue_wins', 0)
        cl = ev_data.get('continue_losses', 0)

        total_folds = fc + fw
        total_continues = cw + cl

        print(f"\n  ── DECISION QUALITY ──")
        if total_folds > 0:
            fold_acc = fc / total_folds
            print(f"  Fold accuracy:      {fold_acc:>6.1%}   "
                  f"({fc}/{total_folds} — model folds where player actually lost)")
        if total_continues > 0:
            call_eff = cw / total_continues
            print(f"  Calling efficiency: {call_eff:>6.1%}   "
                  f"({cw}/{total_continues} — model continues and player won)")

        # ── EV by player quality ──────────────────────────────────────
        evq = ev_data.get('ev_by_quality', {})
        has_evq = any(len(evq.get(q, {}).get('ev_diffs', [])) > 0
                      for q in ('good', 'bad'))
        if has_evq:
            print(f"\n  ── EV BY PLAYER QUALITY ──")
            print(f"  {'Quality':<10} {'N':>6}  {'Player$':>10}  {'Model$':>10}  "
                  f"{'EV diff':>10}  {'BB/hand':>8}")
            print('  ' + '-' * 60)
            for q_name in ['good', 'bad', 'unknown']:
                qd = evq.get(q_name, {})
                diffs = qd.get('ev_diffs', [])
                if not diffs:
                    continue
                n = len(diffs)
                p_total = sum(qd['player_nets'])
                m_total = sum(qd['model_nets'])
                ev_d = sum(diffs)
                bb_h = (ev_d / n / 0.05)
                print(f"  {q_name.capitalize():<10} {n:>6}  ${p_total:>+9.2f}  "
                      f"${m_total:>+9.2f}  ${ev_d:>+9.2f}  {bb_h:>+7.2f}")

        # Per-hand EV
        ev_per_hand_sd = ev_diff_total / total_sp
        bb_per_hand_sd = ev_per_hand_sd / _BB_DOLLARS

        # Project to all-hands edge: showdown hands are a subset, and
        # for non-showdown hands EV diff is structurally 0 (no hole cards
        # to query the model). So the population edge is scaled down by
        # the showdown fraction. This is a more honest table-edge proxy.
        all_6p = parse_stats.get('all_6p_hands') or 0
        sd_count = parse_stats.get('filtered_hands') or 0
        if all_6p > 0 and sd_count > 0:
            sd_fraction = sd_count / all_6p
            bb_per_hand_proj = bb_per_hand_sd * sd_fraction
        else:
            sd_fraction = None
            bb_per_hand_proj = None

        print(f"\n  ── EV HEADLINE ──")
        print(f"  Showdown-only edge:  {bb_per_hand_sd:>+.2f} BB/hand  "
              f"({bb_per_hand_sd * 100:>+.1f} BB/100)")
        print(f"    (per showdown player-hand; aggressive models under-credited)")
        if bb_per_hand_proj is not None:
            print(f"  All-hands projection: {bb_per_hand_proj:>+.2f} BB/hand  "
                  f"({bb_per_hand_proj * 100:>+.1f} BB/100)")
            print(f"    (scaled by showdown fraction {sd_fraction:.1%}; "
                  f"lower bound on table edge)")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print('  ' + '-' * (W - 4))
    if p_actual:
        avg_p = np.mean(p_actual)
        agree = agreement.get('overall', {})
        rate = agree.get('match', 0) / max(agree.get('total', 1), 1)
        # Tightened thresholds. Showdown-player agreement is biased low for
        # tight folders, so 50% is a meaningful "reasonable" floor and 60%
        # is "good". Old 35% threshold mainly checked "not uniform random".
        if rate > 0.60 and avg_p > 0.30:
            verdict = "Model plays strong poker vs NL5 field"
        elif rate > 0.50 and avg_p > 0.25:
            verdict = "Model plays reasonable poker vs NL5 field"
        elif rate > 0.40:
            verdict = "Model shows basic understanding but differs from NL5"
        else:
            verdict = "Model strategy diverges significantly from NL5 play"
        print(f"  Verdict: {verdict}")
        print(f"  (agreement={rate:.1%}, avg_P_actual={avg_p:.3f})")
    print('=' * W)
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def _parse_incremental(hh_dir: Path, num_players: int,
                       max_showdown: int, no_cache: bool):
    """Parse HH files incrementally with early stopping and caching.

    Returns (all_6p_hands, showdown_hands, total_hands, file_count).
    Parses files one by one, collecting 6p hands for population stats
    and showdown hands for model comparison. Stops early once we have
    enough showdown hands (but always parses enough for population stats).
    """
    import time
    from src.hand_history.parser import parse_file

    cache_dir = Path(__file__).parent.parent / 'data' / '.hh_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)

    hh_files = sorted(hh_dir.rglob('*.txt'))
    file_count = len(hh_files)
    print(f'  Found {file_count} .txt files')

    if file_count == 0:
        return [], [], 0, 0

    # Simple cache key: dir path + file count + total size
    total_size = sum(f.stat().st_size for f in hh_files[:100])  # sample for speed
    cache_key = f'{hh_dir}|{file_count}|{total_size}|np{num_players}|max{max_showdown}'
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]
    cache_file = cache_dir / f'parsed_{cache_hash}.pkl'

    if cache_file.exists() and not no_cache:
        print(f'  Loading cached data from {cache_file.name}...')
        try:
            with open(cache_file, 'rb') as cf:
                data = pickle.load(cf)
            print(f'  Loaded: {len(data["all_6p"])} 6p hands, '
                  f'{len(data["showdown"])} showdown, '
                  f'{data["total_hands"]} total parsed')
            return data['all_6p'], data['showdown'], data['total_hands'], file_count
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f'  Cache corrupt ({e}); deleting and re-parsing.')
            try:
                cache_file.unlink()
            except OSError:
                pass

    # Parse incrementally
    all_6p = []
    showdown = []
    total_hands = 0
    files_parsed = 0
    t0 = time.time()

    # We need at least min_6p_for_pop hands for population stats,
    # and max_showdown showdown hands for model comparison.
    # Parse until we have enough of both.
    min_6p_for_pop = max(max_showdown * 5, 50000) if max_showdown > 0 else 0

    for fi, f in enumerate(hh_files):
        try:
            hands = parse_file(str(f))
        except Exception as e:
            continue

        total_hands += len(hands)
        files_parsed += 1

        for h in hands:
            if (abs(h.big_blind_dollars - 0.05) < 0.001
                    and h.num_active_players == num_players):
                all_6p.append(h)
                if h.has_showdown:
                    showdown.append(h)

        # Progress every 1000 files
        if files_parsed % 1000 == 0:
            elapsed = time.time() - t0
            rate = files_parsed / elapsed
            remaining = (file_count - files_parsed) / rate if rate > 0 else 0
            print(f'  [{files_parsed}/{file_count}] '
                  f'{len(all_6p)} 6p, {len(showdown)} showdown  '
                  f'({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)')

        # Early stop if we have enough
        if max_showdown > 0:
            have_enough_showdown = len(showdown) >= max_showdown
            have_enough_pop = len(all_6p) >= min_6p_for_pop
            if have_enough_showdown and have_enough_pop:
                print(f'  Early stop at file {files_parsed}/{file_count}: '
                      f'collected {len(showdown)} showdown + {len(all_6p)} 6p hands')
                break

    elapsed = time.time() - t0
    print(f'  Parsed {files_parsed} files ({total_hands} hands) in {elapsed:.1f}s')

    # Trim showdown to requested amount (random sample for unbiased stats)
    if max_showdown > 0 and len(showdown) > max_showdown:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(showdown), max_showdown, replace=False)
        showdown = [showdown[i] for i in sorted(indices)]

    # Cache results
    data = {
        'all_6p': all_6p,
        'showdown': showdown,
        'total_hands': total_hands,
    }
    try:
        tmp_file = cache_file.with_suffix(f'.pkl.tmp.{os.getpid()}')
        with open(tmp_file, 'wb') as cf:
            pickle.dump(data, cf, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_file, cache_file)
        cache_mb = cache_file.stat().st_size / 1e6
        print(f'  Cached to {cache_file.name} ({cache_mb:.0f} MB)')
    except Exception as e:
        print(f'  Warning: cache save failed: {e}')

    return all_6p, showdown, total_hands, file_count


def main():
    parser = argparse.ArgumentParser(
        description='Validate Deep CFR model against real NL5 hand histories'
    )
    parser.add_argument('--model', type=str, default='',
                        help='Path to .pt checkpoint file (single model)')
    parser.add_argument('--model-facing-bet', type=str, default='',
                        help='Split model: facing bet situations')
    parser.add_argument('--model-checked-to', type=str, default='',
                        help='Split model: checked to situations')
    parser.add_argument('--model-facing-bet-3bet', type=str, default='',
                        help='Split model: facing bet in 3bet pots (>12BB)')
    parser.add_argument('--reraise-classifier', type=str, default='',
                        help='Re-raise binary classifier model (.pt)')
    parser.add_argument('--reraise-threshold', type=float, default=0.25,
                        help='Min P(raise) to upgrade call→raise (default 0.25)')
    parser.add_argument('--no-raise-randomize', action='store_true',
                        help='Disable 80/20 raise-size randomization (always pick argmax raise).')
    parser.add_argument('--hh-dir', type=str,
                        default='data/poker_archive/',
                        help='Directory with PokerStars HH files')
    parser.add_argument('--strategy', type=str, default='avg',
                        choices=['avg', 'current'],
                        help='Strategy to evaluate: avg (default) or current')
    parser.add_argument('--max-hands', type=int, default=0,
                        help='Limit showdown hands for model comparison (0=all)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Force re-parse, ignore cached data')
    parser.add_argument('--exploit', type=float, default=0.0,
                        help='Exploit factor 0.0-1.0 (0=GTO, 1=max exploit vs fish)')
    parser.add_argument('--action-sample-cap', type=float, default=0.0,
                        help='Threshold-sample mode: when > 0, sample from '
                             'model strategy restricted to legal actions with '
                             'prob >= cap (renormalized). Replaces argmax in '
                             'analyze_decision_points / compute_showdown_ev. '
                             '0 = argmax (default).')
    parser.add_argument('--sample-seed', type=int, default=0,
                        help='RNG seed for threshold sampling. Same seed → '
                             'same trajectory of sampled actions (reproducible).')
    parser.add_argument('--no-preflop-ranges', action='store_true',
                        help='Disable preflop range override (default: ranges enabled)')
    parser.add_argument('--use-preflop-lookup', action='store_true',
                        help='Use data-driven preflop lookup (data/preflop_lookup.sqlite) '
                             'instead of static PreflopRanges. Falls back to ranges if DB '
                             'missing. Lets you compare model+lookup vs model+static.')
    parser.add_argument('--preflop-db-path', type=str,
                        default='data/preflop_lookup.sqlite',
                        help='Path to PreflopLookup SQLite DB (only used with '
                             '--use-preflop-lookup)')
    parser.add_argument('--supervised', action='store_true',
                        help='Load as supervised model (StrategyNetwork) instead of Deep CFR')
    parser.add_argument('--blend', type=str, default=None,
                        help='Deep CFR model for fold decisions (blended: fold from CFR, call/raise from --model)')
    parser.add_argument('--fold-blend', type=float, default=1.0,
                        help='Fold blend factor 0.0-1.0 (default: 1.0)')
    parser.add_argument('--min-fold-weight', type=float, default=0.0,
                        help='Minimum CFR fold influence 0.0-1.0 (use 0.3 for ratio_0.00)')
    parser.add_argument('--legacy', action='store_true',
                        help='Force legacy obs format (3-dim game_state, for old models)')
    parser.add_argument('--game-players', type=int, default=0,
                        help='Override game player count (e.g. use 6p model on 4p hands)')
    parser.add_argument('--behavioral-stats', action='store_true',
                        help='Append behavioral stats to observations (for models trained with them)')
    parser.add_argument('--betting-line', action='store_true',
                        help='Append 3-dim betting-line features (requires behavioral stats)')
    parser.add_argument('--threat-range-equity', action='store_true',
                        help='Append equity_vs_threat_range feature (+1 dim, '
                             'requires behavioral stats). Set for models trained '
                             'with --threat-range-equity. Auto-detected by obs size.')
    parser.add_argument('--fold-equity', type=float, default=_DEFAULT_P_FOLD,
                        help=f'Assumed opponent fold probability when model raises '
                             f'where NL5 player did not (default {_DEFAULT_P_FOLD}). '
                             f'Set to 0 to disable raise counterfactual.')
    parser.add_argument('--continuation-equity', type=float,
                        default=_DEFAULT_CONTINUATION_EQUITY,
                        help=f'Fraction of model raise extra-invest recovered '
                             f'through showdown equity when opponent calls '
                             f'(default {_DEFAULT_CONTINUATION_EQUITY}).')
    parser.add_argument('--good-players', type=str, default='',
                        help='Path to good players list (for split metrics)')
    parser.add_argument('--bad-players', type=str, default='',
                        help='Path to bad players list (for split metrics)')
    args = parser.parse_args()

    if args.betting_line and not args.behavioral_stats:
        args.behavioral_stats = True

    game_players_override = args.game_players if args.game_players > 0 else None

    config = load_config(str(Path(__file__).parent.parent / 'configs/training_config.json'))

    # ── Split model mode (uses DecisionEngine — same logic as runtime agent) ──
    if args.model_facing_bet and args.model_checked_to:
        from src.decision_engine import DecisionEngine, DecisionConfig, SpotContext

        dc = DecisionConfig(
            reraise_classifier_path=getattr(args, 'reraise_classifier', ''),
            reraise_threshold=getattr(args, 'reraise_threshold', 0.25),
            raise_randomize=not args.no_raise_randomize,
            use_preflop_ranges=False,  # validation doesn't use preflop ranges
        )
        _de = DecisionEngine(dc)
        _de.load_models(
            facing_bet=args.model_facing_bet,
            checked_to=args.model_checked_to,
            facing_bet_3bet=args.model_facing_bet_3bet or '',
            reraise=getattr(args, 'reraise_classifier', ''),
        )
        print(f'  DecisionEngine: {_de.info()}')

        class _DecisionEngineWrapper:
            """Wraps DecisionEngine to match the get_strategy(obs, pidx, legal) interface."""
            def __init__(self, engine):
                self._engine = engine
                self.iteration = 0
                self.obs_size = engine._obs_size_fb
                self.action_size = 5
                self.num_players = 6
                # Dummy attributes for validate_vs_hh compatibility
                self.strategy_networks = [None] * 6
                self.advantage_networks = [None] * 6
            def get_strategy(self, obs, pidx, legal):
                pot_odds = obs[10] if len(obs) > 10 else 0
                facing_bet = pot_odds > 0.001
                pot_bb = obs[8] * 600 if len(obs) > 8 else 0
                ctx = SpotContext(
                    obs=obs, legal_actions=legal,
                    facing_bet=facing_bet, street='postflop',
                    pot_bb=pot_bb)
                return self._engine.get_strategy(ctx)

        deep_cfr = _DecisionEngineWrapper(_de)
        num_players = 6
        legacy = False
        args.behavioral_stats = True
        print(f'Split model via DecisionEngine (same pipeline as runtime agent)')

    elif not args.model:
        print('Error: --model or (--model-facing-bet + --model-checked-to) required')
        sys.exit(1)

    else:
        model_path = Path(args.model)
        if not model_path.exists():
            print(f'Model not found: {model_path}')
            sys.exit(1)

    hh_dir = Path(args.hh_dir)
    if not hh_dir.exists():
        print(f'HH directory not found: {hh_dir}')
        sys.exit(1)

    # ── Standard single-model loading (skipped if split model) ──
    _split_mode = bool(args.model_facing_bet and args.model_checked_to)
    if _split_mode:
        pass  # deep_cfr already set above
    elif args.blend:
        blend_path = Path(args.blend)
        if not blend_path.exists():
            print(f'Blend model not found: {blend_path}')
            sys.exit(1)
        num_players = game_players_override or 6
        legacy = False  # supervised model uses current obs format
        deep_cfr = BlendedModel(
            supervised_path=str(model_path),
            cfr_path=str(blend_path),
            config=config,
            num_players=num_players,
            fold_blend=args.fold_blend,
            legacy_cfr=args.legacy,
            min_fold_weight=args.min_fold_weight,
        )
        print(f'Loaded blended model (iteration {deep_cfr.iteration})')
    elif args.supervised:
        # Load supervised model (StrategyNetwork checkpoint)
        from src.training.deep_cfr import StrategyNetwork
        from train_supervised import SupervisedModelWrapper
        ckpt = torch.load(str(model_path), map_location='cpu')
        model_np = ckpt.get('num_players', 6)
        num_players = game_players_override or model_np
        obs_size = ckpt['obs_size']
        action_size = ckpt['action_size']
        hidden_layers = ckpt['hidden_layers']
        dropout = ckpt.get('dropout', 0.1)
        model = StrategyNetwork(obs_size, action_size, hidden_layers, dropout)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        deep_cfr = SupervisedModelWrapper(
            model, obs_size, action_size, num_players, 'cpu',
            iteration=ckpt.get('epoch', 0))
        print(f'Auto-detected: {model_np} players' +
              (f' (game: {num_players}p)' if num_players != model_np else ''))
        print(f'Loaded supervised model: {model_path.name}  (epoch {deep_cfr.iteration})')
    else:
        model_np, legacy = _autodetect_players(model_path, force_legacy=args.legacy)
        num_players = game_players_override or model_np
        print(f'Auto-detected: {model_np} players' +
              (f' (game: {num_players}p)' if num_players != model_np else '') +
              (' (legacy obs)' if legacy else ''))

        # Auto-detect feature flags from checkpoint obs size. Always runs —
        # the model's input dim is ground truth; explicit flags are augmented
        # to match. Catches the common case where caller passes --betting-line
        # for a 54-dim model and forgets --threat-range-equity.
        try:
            _ckpt = torch.load(str(model_path), map_location='cpu')
            _input_sz = _ckpt['advantage_networks'][0]['network.0.weight'].shape[1]
            if _input_sz >= 50: args.behavioral_stats = True
            if _input_sz >= 53: args.betting_line = True
            if _input_sz >= 54: args.threat_range_equity = True
            print(f'Auto-detected obs flags: bh={args.behavioral_stats} '
                  f'bl={args.betting_line} threat={args.threat_range_equity} '
                  f'(input_size={_input_sz})')
        except Exception:
            pass

        deep_cfr = _load_model(model_path, config, num_players, legacy,
                               model_num_players=model_np if num_players != model_np else None,
                               behavioral_stats=args.behavioral_stats,
                               betting_line=args.betting_line,
                               threat_range_equity=args.threat_range_equity)
        for net in deep_cfr.advantage_networks + deep_cfr.strategy_networks:
            net.eval()
        print(f'Loaded model: {model_path.name}  (iteration {deep_cfr.iteration})')

    use_bh = args.behavioral_stats
    if use_bh:
        print(f'Behavioral stats: ENABLED (+{BEHAVIORAL_STATS_DIM} dims)')

    # Load player quality lists
    good_players = set()
    bad_players = set()
    if args.good_players and Path(args.good_players).exists():
        with open(args.good_players) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    good_players.add(line)
        print(f'Good players loaded: {len(good_players)}')
    if args.bad_players and Path(args.bad_players).exists():
        with open(args.bad_players) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    bad_players.add(line)
        print(f'Bad players loaded: {len(bad_players)}')

    # Parse hand histories (incremental with early stopping)
    print(f'\nParsing hand histories from {hh_dir} (recursive)...')
    all_6p, showdown, total_hands, file_count = _parse_incremental(
        hh_dir, num_players, args.max_hands, args.no_cache)

    print(f'\n  Total hands parsed:    {total_hands}')
    print(f'  {num_players}-player hands:      {len(all_6p)}')
    print(f'  {num_players}-player showdown:    {len(showdown)}'
          f'  ({len(showdown)/max(total_hands,1)*100:.1f}%)')

    if not showdown:
        print('No qualifying hands found. Check filters.')
        sys.exit(1)

    # Compute NL5 population stats from ALL 6-player hands
    print(f'\nComputing NL5 population stats from {len(all_6p)} hands...')
    nl5_pop_pos, nl5_action_counts = compute_nl5_population_stats(all_6p, num_players)

    # Build decision points (showdown hands only)
    print(f'Building observations from {len(showdown)} showdown hands...')
    builder = ObservationBuilder(num_players=num_players, legacy_game_state=legacy)
    all_decision_points = []
    hands_with_dp = 0

    for i, hand in enumerate(showdown):
        dps = builder.build_decision_points(
            hand, behavioral_stats=use_bh,
            threat_range_equity=args.threat_range_equity)
        if dps:
            all_decision_points.extend(dps)
            hands_with_dp += 1
        if (i + 1) % 10000 == 0:
            print(f'  [{i+1}/{len(showdown)}] {len(all_decision_points)} decision points...')

    print(f'  Hands with decision points: {hands_with_dp}')
    print(f'  Total decision points: {len(all_decision_points)}')

    if not all_decision_points:
        print('No decision points extracted. Check parser/obs_builder.')
        sys.exit(1)

    # Load preflop strategy: either static PreflopRanges or the data-driven
    # PreflopLookup (with PreflopRanges as in-process fallback). The
    # downstream callers expect get_strategy(...) to return an ndarray, so
    # we wrap PreflopLookup (which returns a tuple) in a thin adapter.
    if args.no_preflop_ranges:
        pf_ranges = None
    elif args.use_preflop_lookup:
        try:
            from src.preflop_lookup import PreflopLookup
            from pathlib import Path as _Path
            if not _Path(args.preflop_db_path).exists():
                print(f'Preflop lookup DB not found at {args.preflop_db_path} — '
                      f'falling back to static ranges')
                pf_ranges = PreflopRanges(num_players=num_players)
                print('Preflop: GTO ranges (locked, --use-preflop-lookup fell through)')
            else:
                _fallback = PreflopRanges(num_players=num_players)
                _lookup = PreflopLookup(args.preflop_db_path, fallback=_fallback)

                class _LookupAdapter:
                    """Wrap PreflopLookup so get_strategy returns just the
                    strategy ndarray (drops the custom_size_bb side-channel),
                    matching the legacy PreflopRanges interface that callers
                    in this script expect."""
                    def __init__(self, lookup):
                        self._lookup = lookup
                    def get_strategy(self, *args, **kwargs):
                        result = self._lookup.get_strategy(*args, **kwargs)
                        if isinstance(result, tuple):
                            return result[0]
                        return result

                pf_ranges = _LookupAdapter(_lookup)
                print(f'Preflop: data-driven LOOKUP from {args.preflop_db_path} '
                      f'({_lookup.num_spots} exact spots, '
                      f'{len(_lookup._level_no_nta)} L1 keys, '
                      f'{len(_lookup._level_no_stack)} L2 keys)')
        except Exception as e:
            print(f'PreflopLookup load failed: {e} — falling back to static ranges')
            pf_ranges = PreflopRanges(num_players=num_players)
            print('Preflop: GTO ranges (locked, --use-preflop-lookup errored)')
    else:
        pf_ranges = PreflopRanges(num_players=num_players)
        print('Preflop: GTO ranges (locked)')

    # Run analysis
    exploit_str = f', exploit={args.exploit}' if args.exploit > 0 else ''
    print(f'\nAnalyzing model strategy (mode={args.strategy}{exploit_str})...')

    if args.exploit > 0:
        refiner = PlaytimeStrategy(
            deep_cfr, num_players=num_players,
            temperature=1.0, allin_penalty=False,
            exploit_factor=args.exploit,
        )
        sfn = lambda obs, idx, lg: refiner.get_strategy(obs, idx, lg)
    elif args.strategy == 'current':
        sfn = lambda obs, idx, lg: _current_strategy(deep_cfr, obs, idx, lg)
    else:
        sfn = lambda obs, idx, lg: _avg_strategy(deep_cfr, obs, idx, lg)

    # ── Threshold sampling wrapper ──────────────────────────────────────
    # When --action-sample-cap > 0, replace the model's strategy with a
    # one-hot at a sampled action. The sample is drawn from the model's
    # distribution restricted to legal actions with prob ≥ cap (renormalized).
    # Downstream argmax() in analyze_decision_points / compute_showdown_ev
    # picks the sampled action — equivalent to making the model sample
    # instead of arg-maxing. Falls back to argmax when no legal action
    # meets the cap.
    if args.action_sample_cap > 0:
        cap = float(args.action_sample_cap)
        rng = np.random.default_rng(args.sample_seed)
        _base_sfn = sfn

        def sfn(obs, idx, lg):
            strat = np.asarray(_base_sfn(obs, idx, lg), dtype=np.float64)
            if strat.size == 0:
                return strat
            # Mask to legal actions ≥ cap
            kept_idx = [a for a in lg if a < len(strat) and strat[a] >= cap]
            if not kept_idx:
                return strat  # fallback: original strategy (argmax-equivalent)
            probs = np.array([strat[a] for a in kept_idx])
            s = probs.sum()
            if s <= 0:
                return strat
            probs = probs / s
            sampled = int(rng.choice(kept_idx, p=probs))
            one_hot = np.zeros_like(strat, dtype=np.float32)
            one_hot[sampled] = 1.0
            return one_hot

        print(f'  Threshold sampling enabled: cap={cap:.2f}, seed={args.sample_seed}')

    metrics = analyze_decision_points(
        deep_cfr, all_decision_points, sfn, num_players,
        preflop_ranges=pf_ranges,
        good_players=good_players,
        bad_players=bad_players)

    # Override NL5 position stats with population-level data
    metrics['nl5_pos'] = nl5_pop_pos

    # Showdown EV analysis
    print(f'Computing showdown EV analysis...')
    ev_data = compute_showdown_ev(
        showdown, all_decision_points, sfn, deep_cfr, num_players,
        preflop_ranges=pf_ranges,
        good_players=good_players,
        bad_players=bad_players,
        p_fold_to_raise=args.fold_equity,
        continuation_equity=args.continuation_equity)
    metrics['showdown_ev'] = ev_data

    parse_stats = {
        'files': file_count,
        'total_hands': total_hands,
        'all_6p_hands': len(all_6p),
        'filtered_hands': len(showdown),
    }

    print_report(metrics, parse_stats, deep_cfr, len(all_decision_points))


if __name__ == '__main__':
    main()
