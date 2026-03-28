import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gui import PokerGUI
from src.training import DeepCFR, ModelAdapter
from src.training.deep_cfr import StrategyNetwork
from src.engine import FastEnhancedPokerEngine
from src.search import DepthLimitedSearch, PlaytimeStrategy, MCRolloutSearch
from src.preflop_ranges import PreflopRanges
from src.blended_model import BlendedModel
from src.hand_history.obs_builder import BEHAVIORAL_STATS_DIM, _bh_compute
import numpy as np
import torch
from collections import defaultdict


# ──────────────────────────────────────────────────────────────────────────────
# Live behavioral stats tracker (mirrors obs_builder._bh_track_action)
# ──────────────────────────────────────────────────────────────────────────────

# Action ID to action type name
_ACTION_TYPE = {0: 'fold', 1: 'call', 2: 'raise', 3: 'raise', 4: 'all_in'}


class LiveBehavioralTracker:
    """Track actions during play and compute behavioral stats."""

    def __init__(self, num_players, engine):
        self.num_players = num_players
        self.engine = engine
        self.reset()

    def reset(self):
        """Reset for a new hand."""
        avg_stack = 100  # default 100bb stacks
        try:
            total = sum(
                self.engine.get_player_chips(a)
                for a in self.engine.agents
            )
            avg_stack = max(1, total / self.num_players)
        except Exception:
            pass

        self.bh = {
            'pot_committed': 0,
            'street_bets': defaultdict(int),
            'player_total': defaultdict(int),
            'player_agg': defaultdict(int),
            'player_fold': defaultdict(int),
            'sizing_ratios': [],
            'raise_count': 0,
            'street_actions': [],
            'avg_stack': avg_stack,
        }
        self._last_stage = None

    def track_action(self, agent_name, action_id):
        """Track an action taken during play."""
        # Detect street change
        try:
            stage = self.engine.get_stage()
            if self._last_stage is not None and stage != self._last_stage:
                # Street changed — commit bets to pot
                self.bh['pot_committed'] += sum(self.bh['street_bets'].values())
                self.bh['street_bets'].clear()
                self.bh['street_actions'].clear()
            self._last_stage = stage
        except Exception:
            pass

        atype = _ACTION_TYPE.get(action_id, 'check')
        # Check vs call: action 1 could be check if no bet to face
        if action_id == 1:
            try:
                pid = self.engine.agents.index(agent_name)
                ps = self.engine._state.players_state[pid]
                max_bet = max(p.bet_chips for p in self.engine._state.players_state)
                if ps.bet_chips >= max_bet:
                    atype = 'check'
                else:
                    atype = 'call'
            except Exception:
                atype = 'call'

        is_all_in = (action_id == 4)
        is_agg = atype in ('bet', 'raise') or is_all_in
        name = agent_name

        # Track current street actions
        self.bh['street_actions'].append((name, atype, is_all_in))

        # Per-player counters
        self.bh['player_total'][name] += 1
        if is_agg:
            self.bh['player_agg'][name] += 1
        if atype == 'fold':
            self.bh['player_fold'][name] += 1

        # Sizing ratio for raises
        if atype == 'raise' and action_id in (2, 3, 4):
            try:
                pot = self.engine.get_pot_size()
                if pot > 0:
                    # Estimate raise amount from action type
                    if action_id == 2:
                        ratio = 0.5  # half pot
                    elif action_id == 3:
                        ratio = 1.0  # pot
                    else:
                        # All-in: estimate from stack/pot
                        pid = self.engine.agents.index(name)
                        stake = self.engine._state.players_state[pid].stake
                        ratio = stake / max(pot, 1)
                    self.bh['sizing_ratios'].append((name, ratio))
            except Exception:
                pass

        # Raise count
        if atype == 'raise':
            self.bh['raise_count'] += 1

        # Bet tracking (approximate using pot changes)
        if atype == 'call':
            try:
                pid = self.engine.agents.index(name)
                ps = self.engine._state.players_state[pid]
                max_bet = max(p.bet_chips for p in self.engine._state.players_state)
                call_amt = max_bet - ps.bet_chips
                self.bh['street_bets'][name] += max(call_amt, 0)
            except Exception:
                pass
        elif atype == 'raise':
            try:
                pid = self.engine.agents.index(name)
                ps = self.engine._state.players_state[pid]
                pot = self.engine.get_pot_size()
                if action_id == 2:
                    amt = int(pot / 2)
                elif action_id == 3:
                    amt = pot
                else:
                    amt = ps.stake
                self.bh['street_bets'][name] = max(
                    self.bh['street_bets'].get(name, 0), amt)
            except Exception:
                pass

    def compute(self, hero_name):
        """Compute 8-dim behavioral stats for the given hero."""
        return _bh_compute(self.bh, hero_name)


# ──────────────────────────────────────────────────────────────────────────────
# Strong-hand fold guard
# ──────────────────────────────────────────────────────────────────────────────
# Obs layout (6p): Position(8) + PotOdds(3) + HandStrength(5) + Stack(2) + Equity(8)
# made_hand_cat at index 19:  0.0=high card, 0.125=pair, 0.25=two pair,
#   0.375=trips/set, 0.5=straight, 0.625=flush, 0.75=full house, 0.875=quads

_MADE_HAND_IDX = 19
_FOLD_CAPS = [
    (0.75, 0.00),   # full house+ → never fold
    (0.625, 0.01),  # flush → 1% max fold
    (0.5, 0.02),    # straight → 2% max fold
    (0.375, 0.03),  # set/trips → 3% max fold
    (0.25, 0.05),   # two pair → 5% max fold
]


def apply_strong_hand_guard(strategy, observation, legal_actions):
    """Cap fold probability for strong made hands, redistribute to other actions."""
    if 0 not in legal_actions:
        return strategy

    made_hand = observation[_MADE_HAND_IDX] if len(observation) > _MADE_HAND_IDX else 0.0

    max_fold = 1.0
    for min_cat, cap in _FOLD_CAPS:
        if made_hand >= min_cat - 0.01:
            max_fold = cap
            break

    if max_fold >= 1.0:
        return strategy

    fold_prob = strategy[0]
    if fold_prob <= max_fold:
        return strategy

    excess = fold_prob - max_fold
    strategy = strategy.copy()
    strategy[0] = max_fold

    non_fold_legal = [a for a in legal_actions if a != 0]
    non_fold_sum = sum(strategy[a] for a in non_fold_legal)

    if non_fold_sum > 0:
        for a in non_fold_legal:
            strategy[a] += excess * (strategy[a] / non_fold_sum)
    else:
        for a in non_fold_legal:
            strategy[a] = excess / len(non_fold_legal)

    return strategy


def _make_engine(num_players: int) -> FastEnhancedPokerEngine:
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


class _SupervisedWrapper:
    """Wraps a supervised StrategyNetwork to look like DeepCFR for the GUI."""

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


def load_model(model_path: str, num_players: int = 6, game_num_players: int = None):
    """Load model. If game_num_players differs from model's player count,
    the wrapper will pad observations to match the model's expected input."""
    print(f"\nLoading ENHANCED model from: {model_path}")

    # Auto-detect player count from checkpoint input size
    try:
        checkpoint = torch.load(model_path, map_location='cpu')

        # CQL model (from train_cql.py)
        if checkpoint.get('model_type') == 'cql':
            from scripts.train_cql import QNetwork, CQLModelWrapper
            num_players = checkpoint.get('num_players', num_players)
            obs_size = checkpoint['obs_size']
            action_size = checkpoint['action_size']
            hidden = checkpoint.get('hidden_layers', [512, 512, 256])
            dropout = checkpoint.get('dropout', 0.1)
            temperature = checkpoint.get('temperature', 1.0)
            q_net = QNetwork(obs_size, action_size, hidden, dropout)
            q_net.load_state_dict(checkpoint['model_state_dict'])
            q_net.eval()
            game_np = game_num_players or num_players
            engine = _make_engine(game_np)
            wrapper = CQLModelWrapper(
                q_net, obs_size, action_size, game_np, 'cpu',
                iteration=checkpoint.get('epoch', 0),
                temperature=temperature)
            print(f"  CQL model loaded (epoch {checkpoint.get('epoch', '?')}, "
                  f"alpha={checkpoint.get('alpha_cql', '?')}, temp={temperature})")
            return wrapper, engine, game_np

        # Supervised model (from train_supervised.py)
        if 'model_state_dict' in checkpoint and 'advantage_networks' not in checkpoint:
            model_np = checkpoint.get('num_players', num_players)
            game_np = game_num_players or model_np
            print(f"  Supervised model detected (trained {model_np}p"
                  + (f", game {game_np}p" if game_np != model_np else "") + ")")
            engine = _make_engine(game_np)
            obs_size = checkpoint.get('obs_size',
                                      _make_engine(model_np).get_observation_space_size())
            action_size = checkpoint.get('action_size',
                                          _make_engine(model_np).get_action_space_size())
            hidden = checkpoint.get('hidden_layers', [512, 512, 256])
            dropout = checkpoint.get('dropout', 0.1)
            model = StrategyNetwork(obs_size, action_size, hidden, dropout)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            wrapper = _SupervisedWrapper(
                model, obs_size, action_size, game_np,
                checkpoint.get('epoch', 0),
                model_num_players=model_np if game_np != model_np else None)
            print(f"  Supervised model loaded (epoch {checkpoint.get('epoch', '?')})")
            return wrapper, engine, game_np

        # Deep CFR model
        if 'advantage_networks' in checkpoint and len(checkpoint['advantage_networks']) > 0:
            first_layer_weight = checkpoint['advantage_networks'][0]['network.0.weight']
            model_input_size = first_layer_weight.shape[1]

            for test_players in [2, 3, 4, 5, 6, 7, 8, 9]:
                if _make_engine(test_players).get_observation_space_size() == model_input_size:
                    num_players = test_players
                    print(f"  Auto-detected: {num_players} players (obs size: {model_input_size})")
                    break
            else:
                print(f"  WARNING: could not match obs size {model_input_size}, using {num_players} players")
    except Exception as e:
        print(f"  Could not auto-detect player count: {e}")

    engine = _make_engine(num_players)
    observation_size = engine.get_observation_space_size()
    action_size = engine.get_action_space_size()

    print(f"  Enhanced observation size: {observation_size}")
    print(f"  Number of players: {num_players}")

    deep_cfr = DeepCFR(
        observation_size=observation_size,
        action_size=action_size,
        num_players=num_players,
        hidden_layers=[512, 512, 256],
        learning_rate=0.001,
        buffer_size=100000,
        device='cpu'
    )

    try:
        deep_cfr.load_models(model_path)
        print(f"✓ Enhanced model loaded successfully")
        print(f"  Observation size: {observation_size}")
        print(f"  Action size: {action_size}")
        print(f"  Players: {num_players}")
        return deep_cfr, engine, num_players
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None, None, None


def list_available_models():
    models_dir = Path("models")
    if not models_dir.exists():
        print("No models directory found.")
        return []
    
    model_files = list(models_dir.glob("*.pt"))
    if not model_files:
        print("No model files found in models/")
        return []
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files, 1):
        size = model_file.stat().st_size / (1024 * 1024)
        model_type = "enhanced" if "enhanced" in model_file.name else "base"
        print(f"  {i}. {model_file.name} ({size:.1f} MB) [{model_type}]")
    
    return model_files


def main():
    parser = argparse.ArgumentParser(description='Play poker against a trained ENHANCED AI model')
    parser.add_argument('--model', type=str, help='Path to enhanced model file')
    parser.add_argument('--list', action='store_true', help='List available models')
    parser.add_argument('--players', type=int, default=6, help='Number of players fallback if auto-detect fails (default: 6)')
    parser.add_argument('--search', action='store_true', help='Enable depth-limited search for stronger AI play')
    parser.add_argument('--search-depth', type=int, default=2, help='Search depth in traverser plies (default: 2)')
    parser.add_argument('--search-iters', type=int, default=50, help='CFR iterations per search (default: 50)')
    parser.add_argument('--search-rollouts', type=int, default=10, help='Rollouts per action branch (default: 10)')
    parser.add_argument('--no-refine', action='store_true', help='Disable play-time strategy refinement (temperature + all-in penalty)')
    parser.add_argument('--temperature', type=float, default=0.5, help='Strategy temperature (lower=sharper, default: 0.5)')
    parser.add_argument('--no-preflop-ranges', action='store_true', help='Disable GTO preflop range override')
    parser.add_argument('--exploit', type=float, default=0.0, help='Exploit factor 0.0-1.0 (0=GTO, 0.7=good vs fish)')
    parser.add_argument('--blend', type=str, default=None, help='Deep CFR model for fold decisions (blended mode: fold from CFR, call/raise from --model)')
    parser.add_argument('--fold-blend', type=float, default=1.0, help='Fold blend factor 0.0-1.0 (default: 1.0)')
    parser.add_argument('--strong-hand-guard', action='store_true', help='Cap fold probability for strong made hands (two pair+)')
    parser.add_argument('--behavioral-stats', action='store_true',
                        help='Enable behavioral stats (for models trained with 49-dim obs)')
    parser.add_argument('--mc-search', action='store_true',
                        help='Enable Monte Carlo rollout search (works with any model)')
    parser.add_argument('--mc-rollouts', type=int, default=50,
                        help='Rollouts per action for MC search (default: 50)')
    parser.add_argument('--mc-blend', type=float, default=0.3,
                        help='MC search blend: 0=pure network, 1=pure rollout (default: 0.3)')
    parser.add_argument('--mc-temperature', type=float, default=1.0,
                        help='MC search temperature on normalized EVs: lower=greedier (default: 1.0)')

    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        print("Please specify a model with --model or use --list to see available models")
        print("\nExample:")
        print("  python3 scripts/play_gui_vs_enhanced_model.py --model models/final_model_enhanced.pt")
        print("  python3 scripts/play_gui_vs_enhanced_model.py --list")
        print("\n⚠️  This script is for ENHANCED models only!")
        print("    For base models, use: python3 scripts/play_gui_vs_model.py")
        return
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model file not found: {model_path}")
        print("\nUse --list to see available models")
        return
    
    # Blended mode: supervised + CFR
    if args.blend:
        blend_path = Path(args.blend)
        if not blend_path.exists():
            print(f"❌ Blend model not found: {blend_path}")
            return
        config = load_config()
        deep_cfr = BlendedModel(
            supervised_path=str(model_path),
            cfr_path=str(blend_path),
            config=config,
            num_players=args.players,
            fold_blend=args.fold_blend,
        )
        engine = _make_engine(args.players)
        trained_players = args.players
    else:
        deep_cfr, engine, trained_players = load_model(
            str(model_path), args.players, game_num_players=args.players)
        if deep_cfr is None or engine is None:
            print("\n⚠️  If you're trying to load a BASE model, use:")
            print("    python3 scripts/play_gui_vs_model.py --model", args.model)
            return

    # Set up search if requested
    search = None
    if args.mc_search:
        search = MCRolloutSearch(
            deep_cfr,
            num_rollouts=args.mc_rollouts,
            blend=args.mc_blend,
            temperature=args.mc_temperature,
            obs_pad=BEHAVIORAL_STATS_DIM if args.behavioral_stats else 0,
        )
        print(f"\n  MC rollout search enabled: rollouts={args.mc_rollouts}, "
              f"blend={args.mc_blend}, temperature={args.mc_temperature}")
    elif args.search:
        search = DepthLimitedSearch(
            deep_cfr,
            num_rollouts=args.search_rollouts,
            cfr_iters=args.search_iters,
            max_depth=args.search_depth,
        )
        print(f"\n  CFR search enabled: depth={args.search_depth}, "
              f"iters={args.search_iters}, rollouts={args.search_rollouts}")

    # Set up play-time strategy refinement (on by default)
    refiner = None
    if not args.no_refine:
        refiner = PlaytimeStrategy(
            deep_cfr,
            num_players=trained_players,
            temperature=args.temperature,
            allin_penalty=True,
            exploit_factor=args.exploit,
        )
        exploit_str = f", exploit={args.exploit}" if args.exploit > 0 else ""
        print(f"  Strategy refinement: temperature={args.temperature}, all-in penalty=ON{exploit_str}")

    print("\n" + "="*60)
    print(f"Starting GUI - You vs {trained_players - 1} AI opponents ({trained_players}-player game)")
    if search:
        if args.mc_search:
            print("  (MC rollout search ON — AI will think longer but play stronger)")
        else:
            print("  (Depth-limited CFR search ON — AI will think longer but play stronger)")
    print("="*60)
    print("Controls:")
    print("  - Click buttons to take actions")
    print("  - R: Start new hand")
    print("  - V: Show/Hide opponent cards")
    print("  - Q: Quit")
    print("="*60 + "\n")

    preflop_ranges = None if args.no_preflop_ranges else PreflopRanges(num_players=trained_players)
    if preflop_ranges:
        print(f"  Preflop: GTO ranges (locked)")

    post_hook = apply_strong_hand_guard if args.strong_hand_guard else None
    if args.strong_hand_guard:
        print(f"  Strong-hand guard: ON (two pair+ fold capped)")

    gui = PokerGUI(num_players=trained_players, ai_model=deep_cfr,
                   ai_search=search, ai_strategy_refiner=refiner,
                   preflop_ranges=preflop_ranges,
                   strategy_post_hook=post_hook)
    gui.engine = engine

    # Behavioral stats: wrap process_action to track actions and augment observations
    if args.behavioral_stats:
        print(f"  Behavioral stats: ON (+{BEHAVIORAL_STATS_DIM} dims)")
        bh_tracker = LiveBehavioralTracker(trained_players, engine)

        # Monkey-patch the GUI to inject behavioral stats
        _orig_reset = gui.reset_game
        _orig_process = gui.process_action

        def _patched_reset():
            _orig_reset()
            bh_tracker.engine = gui.engine
            bh_tracker.reset()

        def _patched_process():
            # Wrap get_observation to append behavioral stats
            _orig_get_obs = gui.engine.get_observation

            def _augmented_obs():
                obs = _orig_get_obs()
                if gui.state and gui.state.get('current_player'):
                    hero = gui.state['current_player']
                else:
                    hero = gui.human_player
                bh_feats = bh_tracker.compute(hero)
                return np.concatenate([obs, bh_feats])

            # Wrap step to track all actions (human + AI)
            _orig_step = gui.engine.step

            def _tracked_step(action):
                if gui.state and gui.state.get('current_player'):
                    bh_tracker.track_action(gui.state['current_player'], action)
                return _orig_step(action)

            gui.engine.get_observation = _augmented_obs
            gui.engine.step = _tracked_step
            _orig_process()
            gui.engine.step = _orig_step
            gui.engine.get_observation = _orig_get_obs

        gui.reset_game = _patched_reset
        gui.process_action = _patched_process

    gui.engine.reset()
    gui.run()


if __name__ == "__main__":
    main()
