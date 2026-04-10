#!/usr/bin/env python3
"""
Compare Deep CFR model decisions against Pluribus (superhuman 6-max AI).

Uses the 10,000 published Pluribus hands where ALL hole cards are known
for ALL players — no showdown bias.

Usage:
  python3 scripts/analysis/validate_vs_pluribus.py --model models/checkpoint_iter_2700.pt
  python3 scripts/analysis/validate_vs_pluribus.py --model models/checkpoint_iter_2700.pt --max-hands 2000
"""

import sys
import os
import re
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

_PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'scripts'))

from src.training import DeepCFR
from src.engine import FastEnhancedPokerEngine
from src.agent.obs_builder import ObservationBuilder as AgentObsBuilder

FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN = 0, 1, 2, 3, 4
ACTION_NAMES = {0: 'fold', 1: 'call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
AGGRESSIVE = {RAISE_HALF, RAISE_POT, ALL_IN}

# ─── Pluribus hand parser ────────────────────────────────────────────────

@dataclass
class PluribusAction:
    player: str
    action_type: str  # fold, call, check, raise, allin
    amount: float = 0.0
    street: str = 'preflop'

@dataclass
class PluribusHand:
    hand_id: str = ''
    players: List[str] = field(default_factory=list)
    seats: Dict[str, int] = field(default_factory=dict)
    stacks: Dict[str, float] = field(default_factory=dict)
    hole_cards: Dict[str, List[str]] = field(default_factory=dict)
    board: List[str] = field(default_factory=list)
    actions: List[PluribusAction] = field(default_factory=list)
    button_seat: int = 0
    sb: float = 50.0
    bb: float = 100.0


def _parse_cards(card_str: str) -> List[str]:
    """Parse '[4d 5h]' or '4d 5h' into ['4d', '5h']."""
    card_str = card_str.strip('[]')
    return [c.strip() for c in card_str.split() if len(c.strip()) == 2]


def parse_pluribus_file(filepath: str) -> List[PluribusHand]:
    """Parse a Pluribus PokerStars-format file into structured hands."""
    with open(filepath) as f:
        content = f.read()

    hands = []
    raw_hands = content.split('PokerStars Hand #')

    for raw in raw_hands:
        if not raw.strip():
            continue

        hand = PluribusHand()
        lines = raw.strip().split('\n')
        street = 'preflop'

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Header
            m = re.match(r'(\d+):\s+Hold.em No Limit \((\d+)/(\d+)\)', line)
            if m:
                hand.hand_id = m.group(1)
                hand.sb = float(m.group(2))
                hand.bb = float(m.group(3))
                continue

            # Table
            m = re.match(r"Table .+ Seat #(\d+) is the button", line)
            if m:
                hand.button_seat = int(m.group(1))
                continue

            # Seats
            m = re.match(r'Seat (\d+): (.+?) \((\d+) in chips\)', line)
            if m:
                seat = int(m.group(1))
                name = m.group(2)
                stack = float(m.group(3))
                hand.players.append(name)
                hand.seats[name] = seat
                hand.stacks[name] = stack
                continue

            # Hole cards
            m = re.match(r'Dealt to (.+?) \[(.+?)\]', line)
            if m:
                name = m.group(1)
                cards = _parse_cards(m.group(2))
                hand.hole_cards[name] = cards
                continue

            # Streets
            if '*** FLOP ***' in line:
                street = 'flop'
                m = re.search(r'\[(.+?)\]', line)
                if m:
                    hand.board = _parse_cards(m.group(1))
                continue
            if '*** TURN ***' in line:
                street = 'turn'
                m = re.search(r'\] \[(.+?)\]', line)
                if m:
                    hand.board.append(m.group(1).strip())
                continue
            if '*** RIVER ***' in line:
                street = 'river'
                m = re.search(r'\] \[(.+?)\]', line)
                if m:
                    hand.board.append(m.group(1).strip())
                continue
            if '*** SHOW DOWN ***' in line or '*** SUMMARY ***' in line:
                break

            # Actions
            m = re.match(r'(.+?): (folds|checks|calls|bets|raises)(.*)', line)
            if m:
                name = m.group(1)
                act_raw = m.group(2)
                rest = m.group(3).strip()

                if act_raw == 'folds':
                    act_type = 'fold'
                elif act_raw == 'checks':
                    act_type = 'check'
                elif act_raw == 'calls':
                    act_type = 'call'
                elif act_raw in ('bets', 'raises'):
                    act_type = 'raise'
                else:
                    act_type = act_raw

                # Parse amount
                amount = 0.0
                amt_match = re.search(r'(\d+\.?\d*)', rest)
                if amt_match:
                    amount = float(amt_match.group(1))

                # Check for all-in
                if 'all-in' in rest.lower() or 'and is all-in' in line.lower():
                    act_type = 'allin'

                hand.actions.append(PluribusAction(
                    player=name, action_type=act_type,
                    amount=amount, street=street
                ))
                continue

            # Posts
            m = re.match(r'(.+?): posts (small|big) blind (\d+\.?\d*)', line)
            if m:
                continue  # skip blind posts

        if hand.hand_id and hand.actions:
            hands.append(hand)

    return hands


# ─── Model comparison ────────────────────────────────────────────────────

def _pluribus_action_to_idx(act: PluribusAction, pot: float) -> int:
    """Map Pluribus action to our action space."""
    if act.action_type == 'fold':
        return FOLD
    if act.action_type in ('call', 'check'):
        return CALL
    if act.action_type == 'allin':
        return ALL_IN
    if act.action_type == 'raise':
        if pot > 0:
            ratio = act.amount / pot
            if ratio <= 0.6:
                return RAISE_HALF
            else:
                return RAISE_POT
        return RAISE_POT
    return CALL


def _is_aggressive(action_idx: int) -> bool:
    return action_idx in AGGRESSIVE


def main():
    parser = argparse.ArgumentParser(
        description='Compare model vs Pluribus decisions')
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max-hands', type=int, default=0)
    parser.add_argument('--pluribus-dir', type=str,
                        default='data/pluribus_hands/out/')
    args = parser.parse_args()

    # Load model
    from src.utils import load_config
    config = load_config(str(Path(_PROJECT_ROOT) / 'configs/training_config.json'))

    eng = FastEnhancedPokerEngine(
        num_players=6, add_position_features=True, add_pot_odds_features=True,
        add_hand_strength_features=True, add_stack_features=True,
        add_equity_features=True, add_street_features=True,
        add_game_state_features=True, add_betting_history_features=True,
        behavioral_stats=True,
    )
    obs_size = eng.get_observation_space_size()
    act_size = eng.get_action_space_size()

    # Auto-detect hidden layers from checkpoint
    import torch
    ckpt = torch.load(args.model, map_location='cpu')
    # Infer from first advantage network's first layer
    first_w = ckpt['advantage_networks'][0]
    first_key = [k for k in first_w if 'network.0.weight' in k][0]
    first_hidden = first_w[first_key].shape[0]
    # Try common configs
    if first_hidden == 512:
        hidden = [512, 512, 256, 256, 128]
    else:
        hidden = [256, 256, 128]

    deep_cfr = DeepCFR(
        observation_size=obs_size, action_size=act_size,
        num_players=6, hidden_layers=hidden,
        learning_rate=0.001, buffer_size=1000, device='cpu')
    deep_cfr.load_models(args.model)
    deep_cfr.action_size = max(5, deep_cfr.action_size)

    for net in deep_cfr.advantage_networks + deep_cfr.strategy_networks:
        net.eval()

    print(f'Model: {Path(args.model).name} (iter {deep_cfr.iteration})')

    # Parse Pluribus hands
    print(f'Parsing Pluribus hands from {args.pluribus_dir}...')
    all_hands = []
    for f in sorted(Path(args.pluribus_dir).glob('*.txt')):
        all_hands.extend(parse_pluribus_file(str(f)))
    print(f'  Parsed {len(all_hands)} hands')

    if args.max_hands > 0:
        all_hands = all_hands[:args.max_hands]

    # Build obs builder
    obs_builder = AgentObsBuilder(num_players=6)

    # Compare Pluribus decisions vs model
    total_decisions = 0
    agreements = 0
    agreements_by_street = defaultdict(lambda: {'total': 0, 'agree': 0})
    agreements_by_action = defaultdict(lambda: {'total': 0, 'agree': 0})
    pluribus_actions_dist = defaultdict(int)
    model_actions_dist = defaultdict(int)

    # Action type agreement (fold/call/raise)
    type_agree = 0  # fold=fold, call/check=call, any_raise=any_raise
    type_total = 0

    for hand_idx, hand in enumerate(all_hands):
        if hand_idx % 2000 == 0 and hand_idx > 0:
            print(f'  [{hand_idx}/{len(all_hands)}] '
                  f'{agreements}/{total_decisions} agree '
                  f'({100*agreements/max(total_decisions,1):.1f}%)')

        # Find Pluribus in this hand
        if 'Pluribus' not in hand.hole_cards:
            continue

        pluribus_seat = hand.seats.get('Pluribus', -1)
        if pluribus_seat < 0:
            continue

        # Replay actions and compare at each Pluribus decision
        pot = hand.sb + hand.bb
        active = set(hand.players)
        bets = defaultdict(float)  # current round bets
        street = 'preflop'

        # Track board
        board_so_far = []

        for act_idx, act in enumerate(hand.actions):
            if act.player not in active:
                continue

            if act.street != street:
                street = act.street
                bets.clear()
                if street == 'flop':
                    board_so_far = hand.board[:3]
                elif street == 'turn':
                    board_so_far = hand.board[:4]
                elif street == 'river':
                    board_so_far = hand.board[:5]

            # Only evaluate Pluribus decisions
            if act.player != 'Pluribus':
                # Track pot changes
                if act.action_type == 'fold':
                    active.discard(act.player)
                elif act.action_type in ('call', 'check'):
                    pot += act.amount
                elif act.action_type in ('raise', 'allin'):
                    pot += act.amount
                continue

            # This is a Pluribus decision — compare with our model
            pluribus_action = _pluribus_action_to_idx(act, pot)

            try:
                from src.agent.game_state import GameState, PlayerInfo, CardInfo

                hole = hand.hole_cards.get('Pluribus', [])
                if len(hole) < 2:
                    continue

                # Build PlayerInfo for each player
                ps_list = []
                for pname in hand.players:
                    cards = None
                    if pname == 'Pluribus' and len(hole) >= 2:
                        cards = [CardInfo(c[0], c[1]) for c in hole]
                    ps_list.append(PlayerInfo(
                        seat=hand.seats[pname],
                        name=pname,
                        stack=hand.stacks[pname] / hand.bb,
                        is_active=pname in active,
                        has_button=(hand.seats[pname] == hand.button_seat),
                        hole_cards=cards,
                    ))

                board_cards = [CardInfo(c[0], c[1]) for c in board_so_far]

                gs = GameState(
                    players=ps_list,
                    my_seat=hand.seats['Pluribus'],
                    community_cards=board_cards,
                    pot_total=pot / hand.bb,
                    street=street,
                    button_seat=hand.button_seat,
                    small_blind=0.5,
                    big_blind=1.0,
                    small_blind_dollars=hand.sb,
                    big_blind_dollars=hand.bb,
                    legal_actions=[0, 1, 2, 3, 4],
                    num_active_players=len(active),
                )

                obs = obs_builder.build(gs)
                if obs is None:
                    continue

                # Pad to 50 dims if needed (model expects behavioral stats)
                if len(obs) < obs_size:
                    padded = np.zeros(obs_size, dtype=np.float32)
                    padded[:len(obs)] = obs
                    obs = padded

                # Get model strategy
                strategy = deep_cfr.get_strategy(obs, 0, [0, 1, 2, 3])
                model_action = int(np.argmax(strategy[:4]))

            except Exception as e:
                if total_decisions == 0 and hand_idx < 5:
                    import traceback
                    traceback.print_exc()
                continue

            total_decisions += 1
            pluribus_actions_dist[pluribus_action] += 1
            model_actions_dist[model_action] += 1

            # Exact agreement
            if model_action == pluribus_action:
                agreements += 1

            # Type agreement (fold vs call vs raise)
            p_type = 'fold' if pluribus_action == FOLD else ('call' if pluribus_action == CALL else 'raise')
            m_type = 'fold' if model_action == FOLD else ('call' if model_action == CALL else 'raise')
            type_total += 1
            if p_type == m_type:
                type_agree += 1

            agreements_by_street[street]['total'] += 1
            if model_action == pluribus_action:
                agreements_by_street[street]['agree'] += 1

            agreements_by_action[p_type]['total'] += 1
            if p_type == m_type:
                agreements_by_action[p_type]['agree'] += 1

            # Update pot
            if act.action_type == 'fold':
                active.discard('Pluribus')
            elif act.action_type in ('call', 'check'):
                pot += act.amount
            elif act.action_type in ('raise', 'allin'):
                pot += act.amount

    # ── Report ────────────────────────────────────────────────────────
    print()
    print('=' * 70)
    print(f'  PLURIBUS COMPARISON  ({len(all_hands)} hands, '
          f'{total_decisions} decisions)')
    print('=' * 70)

    if total_decisions == 0:
        print('  No decisions compared — check parser')
        return

    print(f'\n  Exact agreement:  {agreements}/{total_decisions} '
          f'({100*agreements/total_decisions:.1f}%)')
    print(f'  Type agreement:   {type_agree}/{type_total} '
          f'({100*type_agree/type_total:.1f}%)')

    print(f'\n  Agreement by street:')
    for st in ['preflop', 'flop', 'turn', 'river']:
        d = agreements_by_street[st]
        if d['total'] > 0:
            pct = 100 * d['agree'] / d['total']
            print(f'    {st:>8}: {d["agree"]:>5}/{d["total"]:<5} ({pct:.1f}%)')

    print(f'\n  Agreement by Pluribus action type:')
    for at in ['fold', 'call', 'raise']:
        d = agreements_by_action[at]
        if d['total'] > 0:
            pct = 100 * d['agree'] / d['total']
            print(f'    {at:>8}: {d["agree"]:>5}/{d["total"]:<5} ({pct:.1f}%)')

    print(f'\n  Action distribution:')
    print(f'    {"Action":>12} {"Pluribus":>10} {"Model":>10}')
    for a in [FOLD, CALL, RAISE_HALF, RAISE_POT, ALL_IN]:
        p_pct = 100 * pluribus_actions_dist[a] / max(total_decisions, 1)
        m_pct = 100 * model_actions_dist[a] / max(total_decisions, 1)
        print(f'    {ACTION_NAMES[a]:>12} {p_pct:>9.1f}% {m_pct:>9.1f}%')

    print('=' * 70)


if __name__ == '__main__':
    main()
