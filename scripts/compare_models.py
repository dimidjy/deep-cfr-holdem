"""
Head-to-head comparison of two poker models.

Plays N hands where both models make decisions on the same game states,
then compares total profit/loss.

Each hand is dealt once. Both models see the same cards and board.
Model A plays seats 0,2,4 — Model B plays seats 1,3,5.
After N/2 hands, seats swap so position bias is neutralized.

Usage:
  python3 scripts/compare_models.py \
    --model-a models/supervised_6p_v6.pt \
    --model-b models/cql_6p_v1.pt \
    --hands 10000
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.engine import FastEnhancedPokerEngine
from src.training.deep_cfr import StrategyNetwork
from src.preflop_ranges import PreflopRanges


def _make_engine(num_players=6):
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


def load_model(path, num_players=6):
    """Load any model type, return (wrapper, obs_size, model_name)."""
    checkpoint = torch.load(path, map_location='cpu')

    if checkpoint.get('model_type') == 'cql':
        from scripts.train_cql import QNetwork, CQLModelWrapper
        obs_size = checkpoint['obs_size']
        action_size = checkpoint['action_size']
        hidden = checkpoint.get('hidden_layers', [512, 512, 256])
        dropout = checkpoint.get('dropout', 0.1)
        temp = checkpoint.get('temperature', 1.0)
        q_net = QNetwork(obs_size, action_size, hidden, dropout)
        q_net.load_state_dict(checkpoint['model_state_dict'])
        q_net.eval()
        model = CQLModelWrapper(q_net, obs_size, action_size, num_players,
                                'cpu', checkpoint.get('epoch', 0), temp)
        name = f"CQL(ep{checkpoint.get('epoch', '?')})"
        return model, obs_size, name

    if 'model_state_dict' in checkpoint and 'advantage_networks' not in checkpoint:
        obs_size = checkpoint.get('obs_size', 41)
        action_size = checkpoint.get('action_size', 5)
        hidden = checkpoint.get('hidden_layers', [512, 512, 256])
        dropout = checkpoint.get('dropout', 0.1)
        net = StrategyNetwork(obs_size, action_size, hidden, dropout)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        model = _SupervisedWrapper(net, obs_size, action_size, num_players,
                                   checkpoint.get('epoch', 0))
        name = f"Supervised(ep{checkpoint.get('epoch', '?')})"
        return model, obs_size, name

    raise ValueError(f"Unknown model type in {path}")


class _SupervisedWrapper:
    def __init__(self, model, obs_size, action_size, num_players, iteration=0):
        self.action_size = action_size
        self.num_players = num_players
        self.iteration = iteration
        self.strategy_networks = [model] * num_players
        self.advantage_networks = [model] * num_players

    def get_strategy(self, observation, player_idx, legal_actions):
        obs_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            logits = self.strategy_networks[player_idx](obs_t).numpy()[0]
        masked = np.full(self.action_size, -1e9, dtype=np.float32)
        for a in legal_actions:
            masked[a] = logits[a]
        masked -= masked.max()
        exp_s = np.exp(masked)
        return exp_s / exp_s.sum()


def play_hand(engine, models, seat_assignment, preflop_ranges, obs_sizes):
    """Play one hand. Returns {agent_name: reward}."""
    engine.reset()

    while not engine.is_done():
        agent = engine.current_agent
        if agent not in engine.agents:
            break
        agent_idx = engine.agents.index(agent)
        legal = engine.get_legal_actions()
        if not legal:
            break

        model_key = seat_assignment[agent_idx]  # 'a' or 'b'
        model = models[model_key]
        obs_size = obs_sizes[model_key]

        obs = engine.get_observation()
        # Pad observation if model expects more dims
        if len(obs) < obs_size:
            obs = np.concatenate([obs, np.zeros(obs_size - len(obs), dtype=np.float32)])

        # Preflop: use GTO ranges
        use_range = False
        try:
            stage = engine.get_stage()
            if stage == 'Preflop' and preflop_ranges:
                cards = engine.get_player_cards(agent)
                if cards and len(cards) >= 2:
                    button_idx = engine._button
                    rel_pos = (agent_idx - button_idx) % len(engine.agents)
                    pot_bb = engine.get_pot_size() / 2.0
                    strategy = preflop_ranges.get_strategy(
                        cards, rel_pos, pot_bb, legal, engine.get_action_space_size())
                    use_range = True
        except Exception:
            pass

        if not use_range:
            strategy = model.get_strategy(obs, agent_idx, legal)

        # Pick best action (greedy)
        best_action = legal[0]
        best_prob = -1
        for a in legal:
            if strategy[a] > best_prob:
                best_prob = strategy[a]
                best_action = a

        engine.step(best_action)

    return engine.get_rewards()


def main():
    parser = argparse.ArgumentParser(description='Compare two poker models head-to-head')
    parser.add_argument('--model-a', type=str, required=True, help='First model path')
    parser.add_argument('--model-b', type=str, required=True, help='Second model path')
    parser.add_argument('--hands', type=int, default=10000, help='Total hands to play')
    parser.add_argument('--num-players', type=int, default=6)
    parser.add_argument('--no-preflop-ranges', action='store_true')
    args = parser.parse_args()

    num_players = args.num_players

    # Load models
    print(f"Loading Model A: {args.model_a}")
    model_a, obs_a, name_a = load_model(args.model_a, num_players)
    print(f"  → {name_a} (obs={obs_a})")

    print(f"Loading Model B: {args.model_b}")
    model_b, obs_b, name_b = load_model(args.model_b, num_players)
    print(f"  → {name_b} (obs={obs_b})")

    models = {'a': model_a, 'b': model_b}
    obs_sizes = {'a': obs_a, 'b': obs_b}

    preflop_ranges = None if args.no_preflop_ranges else PreflopRanges(num_players=num_players)

    engine = _make_engine(num_players)
    half = args.hands // 2

    # Track results
    total_a = 0.0
    total_b = 0.0
    hands_played = 0
    action_counts_a = defaultdict(int)
    action_counts_b = defaultdict(int)

    print(f"\n{'='*60}")
    print(f"  {name_a}  vs  {name_b}")
    print(f"  {args.hands} hands, {num_players} players")
    print(f"  Preflop ranges: {'ON' if preflop_ranges else 'OFF'}")
    print(f"{'='*60}\n")

    t0 = time.time()

    for hand_num in range(args.hands):
        # First half: A=seats 0,2,4  B=seats 1,3,5
        # Second half: swap
        if hand_num < half:
            seat_map = {i: ('a' if i % 2 == 0 else 'b') for i in range(num_players)}
        else:
            seat_map = {i: ('b' if i % 2 == 0 else 'a') for i in range(num_players)}

        rewards = play_hand(engine, models, seat_map, preflop_ranges, obs_sizes)

        for agent_name, reward in rewards.items():
            idx = engine.agents.index(agent_name)
            which = seat_map[idx]
            if which == 'a':
                total_a += reward
            else:
                total_b += reward

        hands_played += 1

        if (hand_num + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (hand_num + 1) / elapsed
            a_per_hand = total_a / (hand_num + 1)
            b_per_hand = total_b / (hand_num + 1)
            print(f"  [{hand_num+1}/{args.hands}] "
                  f"A={total_a:+.1f} ({a_per_hand:+.2f}/hand)  "
                  f"B={total_b:+.1f} ({b_per_hand:+.2f}/hand)  "
                  f"({rate:.0f} hands/s)")

    elapsed = time.time() - t0

    # Results
    a_bb100 = (total_a / hands_played) * 100 / 2  # convert chips to BB/100
    b_bb100 = (total_b / hands_played) * 100 / 2
    diff_bb100 = a_bb100 - b_bb100

    print(f"\n{'='*60}")
    print(f"  RESULTS ({hands_played} hands, {elapsed:.0f}s)")
    print(f"{'='*60}")
    print(f"  Model A ({name_a}):")
    print(f"    Total: {total_a:+.1f} chips  ({a_bb100:+.1f} BB/100)")
    print(f"  Model B ({name_b}):")
    print(f"    Total: {total_b:+.1f} chips  ({b_bb100:+.1f} BB/100)")
    print(f"\n  Difference: {diff_bb100:+.1f} BB/100 in favor of "
          f"{'A (' + name_a + ')' if diff_bb100 > 0 else 'B (' + name_b + ')'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
