#!/usr/bin/env python3
"""Inspect model raw strategy output on ~20 typical poker spots.

Builds synthetic GameState objects for common situations and queries the model
directly (no preflop ranges, no guards, no filters — raw model output only).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from src.agent.obs_builder import ObservationBuilder
from src.agent.game_state import GameState, PlayerInfo, CardInfo

import argparse as _ap
_p = _ap.ArgumentParser(add_help=False)
_p.add_argument('--model', default='models/cql_6p_v11.pt')
_args, _ = _p.parse_known_args()
MODEL_PATH = _args.model
NUM_PLAYERS = 6
ACTION_NAMES = ['Fold', 'Call', 'Raise½', 'RaisePot', 'AllIn']


def make_card(s: str) -> CardInfo:
    return CardInfo(rank=s[0], suit=s[1])


def make_players(my_seat, btn_seat, stacks, bets, active_mask=None, all_in_mask=None):
    """Build 6 players. stacks/bets in BB."""
    players = []
    for i in range(NUM_PLAYERS):
        is_active = active_mask[i] if active_mask else True
        is_all_in = all_in_mask[i] if all_in_mask else False
        players.append(PlayerInfo(
            seat=i, name=f'P{i}', stack=stacks[i],
            is_active=is_active, is_all_in=is_all_in,
            bet_current_round=bets[i],
            total_invested=bets[i],
            has_button=(i == btn_seat),
        ))
    return players


def make_gs(my_seat, btn_seat, hole, board, street, pot, call_amt,
            stacks=None, bets=None, active_mask=None, all_in_mask=None,
            min_raise=2.0):
    if stacks is None:
        stacks = [100.0] * NUM_PLAYERS
    if bets is None:
        bets = [0.0] * NUM_PLAYERS
    if active_mask is None:
        active_mask = [True] * NUM_PLAYERS

    players = make_players(my_seat, btn_seat, stacks, bets, active_mask, all_in_mask)
    players[my_seat].hole_cards = [make_card(c) for c in hole]

    return GameState(
        players=players,
        my_seat=my_seat,
        community_cards=[make_card(c) for c in board],
        pot_total=pot,
        street=street,
        button_seat=btn_seat,
        small_blind=0.5,
        big_blind=1.0,
        legal_actions=[0, 1, 2, 3, 4],
        num_active_players=sum(active_mask),
        call_amount=call_amt,
        min_raise=min_raise,
    )


# ─── Define 20 spots ────────────────────────────────────────────────────────

SPOTS = [
    # ═══════════════════════════════════════════════════════════════════
    # FLOP — Checked to (IP, no bet to face)
    # ═══════════════════════════════════════════════════════════════════
    {
        'name': 'FLOP | AA overpair dry (Ks7d2c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Ah', 'Ad'], board=['Ks', '7d', '2c'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | TPTK (AK on A72r) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Ah', 'Kd'], board=['As', '7d', '2c'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | Set (77 on 7sKd3c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['7h', '7c'], board=['7s', 'Kd', '3c'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | Middle pair (TT on Ks Td 5c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Th', 'Tc'], board=['Ks', 'Td', '5c'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | Nut flush draw (Ah5h on Kh8h3c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Ah', '5h'], board=['Kh', '8h', '3c'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | OESD (JT on 9c8d2h) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Jh', 'Td'], board=['9c', '8d', '2h'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'FLOP | Complete air (72o on AKQ) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['7c', '2d'], board=['Ah', 'Kd', 'Qc'],
            street='flop', pot=7.0, call_amt=0.0, bets=[0]*6),
    },

    # ═══════════════════════════════════════════════════════════════════
    # FLOP — Facing bet (OOP, must decide call/fold/raise)
    # ═══════════════════════════════════════════════════════════════════
    {
        'name': 'FLOP | AA wet (JhTh9c) face ½pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Ah', 'Ad'], board=['Jh', 'Th', '9c'],
            street='flop', pot=10.0, call_amt=3.5, bets=[3.5, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'FLOP | Nut FD (Ah5h on Kh8h3c) face pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Ah', '5h'], board=['Kh', '8h', '3c'],
            street='flop', pot=14.0, call_amt=7.0, bets=[7.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'FLOP | Bottom pair (33 on Ks9d3c) face pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['3h', '3d'], board=['Ks', '9d', '3c'],
            street='flop', pot=14.0, call_amt=7.0, bets=[7.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'FLOP | Air (72o on AKQ) face ½pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['7c', '2d'], board=['Ah', 'Kd', 'Qc'],
            street='flop', pot=10.0, call_amt=3.5, bets=[3.5, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'FLOP | Air (72o on AKQ) face FULL POT bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['7c', '2d'], board=['Ah', 'Kd', 'Qc'],
            street='flop', pot=14.0, call_amt=7.0, bets=[7.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'FLOP | Gutshot (KQ on Jh9d3c) face ½pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Kh', 'Qd'], board=['Jh', '9d', '3c'],
            street='flop', pot=10.0, call_amt=3.5, bets=[3.5, 0, 0, 0, 0, 0]),
    },

    # ═══════════════════════════════════════════════════════════════════
    # TURN — checked to / facing bet
    # ═══════════════════════════════════════════════════════════════════
    {
        'name': 'TURN | Nut flush hit (Ah5h Kh8h3c2h) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Ah', '5h'],
            board=['Kh', '8h', '3c', '2h'], street='turn',
            pot=10.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'TURN | Missed draw (As5s Kh8h3c2d) face pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['As', '5s'],
            board=['Kh', '8h', '3c', '2d'], street='turn',
            pot=16.0, call_amt=8.0, bets=[8.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'TURN | TPTK (AK on A72r 6c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Ah', 'Kd'],
            board=['As', '7d', '2c', '6c'], street='turn',
            pot=14.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'TURN | Overpair (AA on 9c6d3h2s) face ½pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Ah', 'Ad'],
            board=['9c', '6d', '3h', '2s'], street='turn',
            pot=16.0, call_amt=5.0, bets=[5.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'TURN | Air (72o on AKQ5) face ½pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['7c', '2d'],
            board=['Ah', 'Kd', 'Qc', '5s'], street='turn',
            pot=16.0, call_amt=5.0, bets=[5.0, 0, 0, 0, 0, 0]),
    },

    # ═══════════════════════════════════════════════════════════════════
    # RIVER — checked to / facing bet
    # ═══════════════════════════════════════════════════════════════════
    {
        'name': 'RIVER | Full house (77 on 7sKd3c3h9d) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['7h', '7c'],
            board=['7s', 'Kd', '3c', '3h', '9d'], street='river',
            pot=20.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'RIVER | 2nd pair (KQ on Qs9d6c3h2s) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Kh', 'Qd'],
            board=['Qs', '9d', '6c', '3h', '2s'], street='river',
            pot=12.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'RIVER | Bluff catcher A-high (K9632r) face pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Ah', '4d'],
            board=['Kd', '9c', '6h', '3s', '2c'], street='river',
            pot=40.0, call_amt=20.0, bets=[20.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'RIVER | Air (72o on AKQ53) face ⅓pot bet',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['7c', '2d'],
            board=['Ah', 'Kd', 'Qc', '5s', '3h'], street='river',
            pot=15.0, call_amt=5.0, bets=[5.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'RIVER | Air (72o on AKQ53) face OVERBET 2x pot',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['7c', '2d'],
            board=['Ah', 'Kd', 'Qc', '5s', '3h'], street='river',
            pot=50.0, call_amt=40.0, bets=[40.0, 0, 0, 0, 0, 0]),
    },
    {
        'name': 'RIVER | Straight (JT on 9c8d2h7s3c) check to',
        'gs': lambda: make_gs(
            my_seat=0, btn_seat=0, hole=['Jh', 'Td'],
            board=['9c', '8d', '2h', '7s', '3c'], street='river',
            pot=16.0, call_amt=0.0, bets=[0]*6),
    },
    {
        'name': 'RIVER | Top pair (AK on As7d2c6c9h) face ½pot',
        'gs': lambda: make_gs(
            my_seat=2, btn_seat=0, hole=['Ah', 'Kd'],
            board=['As', '7d', '2c', '6c', '9h'], street='river',
            pot=24.0, call_amt=8.0, bets=[8.0, 0, 0, 0, 0, 0]),
    },
]


def load_model():
    """Load model and obs_builder, inferring obs blocks from the checkpoint width."""
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    model_type = checkpoint.get('model_type', 'deep_cfr')
    model_obs_size = checkpoint.get('obs_size', 38)
    model_action_size = checkpoint.get('action_size', 5)

    if 'advantage_networks' in checkpoint:
        adv0 = checkpoint['advantage_networks'][0]
        model_obs_size = adv0['network.0.weight'].shape[1]
        last_w = max((k for k in adv0.keys() if k.endswith('.weight') and adv0[k].dim() == 2),
                     key=lambda k: int(k.split('.')[1]) if k.split('.')[1].isdigit() else -1)
        model_action_size = adv0[last_w].shape[0]

    if model_type in ('cql', 'iql'):
        from scripts.train_cql import QNetwork, CQLModelWrapper
        q_net = QNetwork(
            model_obs_size, model_action_size,
            checkpoint.get('hidden_layers', [512, 512, 256]),
            checkpoint.get('dropout', 0.1))
        q_net.load_state_dict(checkpoint['model_state_dict'])
        q_net.eval()
        model = CQLModelWrapper(
            q_net, model_obs_size, model_action_size,
            NUM_PLAYERS, 'cpu',
            iteration=checkpoint.get('epoch', 0),
            temperature=checkpoint.get('temperature', 1.0))
        extra = model_obs_size - 38
        has_bh = checkpoint.get('behavioral_stats', False) or extra >= 8
        has_evo = extra >= 9
        if has_bh:
            model._behavioral_stats = True
        if has_evo:
            model._equity_vs_odds = True
    else:
        from src.training import DeepCFR
        deep_cfr = DeepCFR(
            observation_size=model_obs_size, action_size=model_action_size,
            num_players=NUM_PLAYERS, hidden_layers=[512, 512, 256],
            learning_rate=0.001, buffer_size=100000, device='cpu')
        deep_cfr.load_models(MODEL_PATH)
        model = deep_cfr
        # Agent obs_builder bundles betting_line with behavioral_stats:
        # 42=base, 53=base+bh+betting_line, 54=base+bh+betting_line+evo
        has_bh = model_obs_size >= 50
        has_evo = model_obs_size >= 54

    obs_builder = ObservationBuilder(NUM_PLAYERS, behavioral_stats=has_bh,
                                      equity_vs_odds=has_evo)

    info = {
        'type': model_type, 'obs_size': model_obs_size,
        'action_size': model_action_size, 'epoch': checkpoint.get('epoch', '?'),
    }
    return model, obs_builder, info


def get_raw_strategy(model, obs, legal, action_size=5):
    """Get raw model strategy (no guards)."""
    raw = model.get_strategy(obs, 0, legal)
    probs = np.zeros(action_size, dtype=np.float32)
    if isinstance(raw, dict):
        for a, p in raw.items():
            probs[a] = p
    elif isinstance(raw, (list, np.ndarray)):
        for j in range(min(len(raw), action_size)):
            probs[j] = float(raw[j])
    return probs


def apply_full_pipeline(model, obs_builder, obs, legal, game_state,
                        temperature=1.0):
    """Apply the decision-pipeline overrides after model inference."""
    try:
        from src.agent.policy_core import (
            _apply_strong_hand_guard, _apply_allin_filter, _apply_call_guard,
        )
        guards_available = True
    except ImportError:
        guards_available = False

    action_size = 5
    strategy = get_raw_strategy(model, obs, legal, action_size)

    if guards_available:
        strategy = _apply_strong_hand_guard(strategy, obs, legal)
        strategy = _apply_allin_filter(strategy, obs, legal)
        strategy = _apply_call_guard(strategy, obs, legal, game_state,
                                      model, obs_builder)

    # Temperature scaling
    if temperature != 1.0 and temperature > 0:
        result = np.zeros_like(strategy)
        probs = np.array([strategy[a] for a in legal])
        probs = np.clip(probs, 1e-8, None)
        log_probs = np.log(probs) / temperature
        log_probs -= log_probs.max()
        exp_probs = np.exp(log_probs)
        exp_probs /= exp_probs.sum()
        for idx, a in enumerate(legal):
            result[a] = exp_probs[idx]
        strategy = result

    return strategy


def format_strategy(probs, legal, action_size=5):
    """Format strategy array for display."""
    return '  '.join(
        f'{probs[j]:>8.1%}' if j in legal else f'{"--":>8}'
        for j in range(action_size))


def main():
    model, obs_builder, info = load_model()
    model_action_size = info['action_size']
    temperature = 1.0  # match agent config

    print(f"Model: {MODEL_PATH}")
    print(f"Type: {info['type']}, obs={info['obs_size']}, actions={model_action_size}, "
          f"epoch={info['epoch']}, temp={temperature}")
    print()

    act_header = '  '.join(f'{n:>8}' for n in ACTION_NAMES)
    hdr = f"{'#':>2} {'Spot':<52}"
    print(f"{hdr} │ {'RAW MODEL':^48} │ {'AFTER GUARDS':^48} │")
    print(f"{'':>2} {'':52} │ {act_header}  {'→':>2} │ {act_header}  {'→':>2} │")
    print("─" * 165)

    for i, spot in enumerate(SPOTS):
        gs = spot['gs']()
        obs = obs_builder.build(gs)
        if obs is None:
            print(f"{i+1:>2} {spot['name']:<52} │ {'[obs build failed]':^48} │")
            continue

        legal = [a for a in gs.legal_actions if a < model_action_size]
        if not legal:
            legal = [a for a in gs.legal_actions if a < len(ACTION_NAMES)]

        # Raw model output
        raw = get_raw_strategy(model, obs, legal, model_action_size)
        raw_best = legal[np.argmax([raw[a] for a in legal])]
        raw_name = ACTION_NAMES[raw_best]

        # Full pipeline (guards + temperature)
        final = apply_full_pipeline(model, obs_builder, obs, legal, gs, temperature)
        fin_best = legal[np.argmax([final[a] for a in legal])]
        fin_name = ACTION_NAMES[fin_best]

        raw_str = format_strategy(raw, legal, model_action_size)
        fin_str = format_strategy(final, legal, model_action_size)

        changed = '◄' if raw_best != fin_best else ' '
        print(f"{i+1:>2} {spot['name']:<52} │ {raw_str} {raw_name:>8} │ {fin_str} {fin_name:>8}{changed}│")

    print()
    print("◄ = guards changed the action vs raw model")
    print(f"Pipeline: model → strong_hand_guard → allin_filter → call_guard → temperature({temperature})")


if __name__ == '__main__':
    main()
