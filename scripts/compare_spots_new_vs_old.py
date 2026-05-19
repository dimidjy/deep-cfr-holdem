"""Compare v55/4100 (new, merged equity) vs iter_4000 (old, equity-vs-random)
on real spots from logs/action_debug.

For each spot:
  - Rebuild obs using merged equity logic (53-dim)
  - Run both models, report strategy
  - Print key obs values (obs[18] hand_strength, equity context)
  - Compare to logged strategy (from the original 54-dim run)

Usage:
    python3 scripts/compare_spots_new_vs_old.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn as nn

from src.agent.game_state import GameState
from src.agent.obs_builder import ObservationBuilder


# Spots to test (from logs/action_debug)
# Each tuple: (file, hero_has_initiative_override, hand_raise_count_override)
# These overrides feed Fix 5 (opp_aggressed_this_hand).
SPOT_FILES = [
    ('logs/action_debug/20260516_204800_792_raise_custom_2.00bb.json',
        None, None),  # flop checked-to, preflop just limps → unclear
    ('logs/action_debug/20260516_204808_469_call.json',
        False, 1),    # villain raised flop now → opp has initiative
    ('logs/action_debug/20260516_204817_764_raise_custom_11.50bb.json',
        False, 2),    # turn checks but opp raised flop earlier
    ('logs/action_debug/20260516_204830_602_all_in.json',
        False, 2),    # river checks but opp raised flop earlier
]

ACTION_NAMES = ['Fold', 'Call', 'Raise½', 'RaisePot', 'AllIn', 'RaiseMin']


def _load_model(path: Path, obs_size: int = 53, act_size: int = 6,
                hidden=[512, 512, 256], dropout=0.15) -> nn.Sequential:
    """Load advantage net for inference. Pads 53→54 etc if needed."""
    ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
    state = ckpt['advantage_networks'][0]
    src_obs = int(state['network.0.weight'].shape[1])
    if src_obs > obs_size:
        # Trim if model is bigger than target (54→53)
        state = dict(state)
        state['network.0.weight'] = state['network.0.weight'][:, :obs_size]
    elif src_obs < obs_size:
        # Pad if model is smaller (rare)
        w = state['network.0.weight']
        pad = torch.zeros(w.shape[0], obs_size - src_obs, dtype=w.dtype)
        state = dict(state)
        state['network.0.weight'] = torch.cat([w, pad], dim=1)

    layers = []
    prev = obs_size
    for h in hidden:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.LeakyReLU(0.01))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, act_size))
    net = nn.Sequential(*layers)
    new_state = {}
    for k, v in state.items():
        nk = k.replace('network.', '')
        if nk in net.state_dict() and net.state_dict()[nk].shape == v.shape:
            new_state[nk] = v
    net.load_state_dict(new_state, strict=False)
    net.eval()
    return net, src_obs


def _regret_match(adv, legal):
    """Strategy from advantage via regret matching (positive regrets only)."""
    pos = np.maximum(adv, 0)
    out = np.zeros_like(adv)
    legal_pos = sum(pos[a] for a in legal)
    if legal_pos > 0:
        for a in legal:
            out[a] = pos[a] / legal_pos
    else:
        for a in legal:
            out[a] = 1.0 / len(legal)
    return out


def _strategy_from_logits(adv, legal):
    return _regret_match(adv, legal)


def main():
    print('Loading models...')
    new_model, new_src = _load_model(Path('models/v55/checkpoint_iter_4100.pt'))
    old_model, old_src = _load_model(Path('models/best/checkpoint_iter_4000.pt'))
    print(f'  v55/4100: src_dim={new_src} (target 53)')
    print(f'  iter_4000: src_dim={old_src} (target 53)')

    builder = ObservationBuilder(num_players=6, pad_to_6p=False,
                                   behavioral_stats=True, equity_vs_odds=True)

    for spot_file, hero_init_override, raise_count_override in SPOT_FILES:
        path = Path(spot_file)
        if not path.exists():
            print(f'\nMISSING: {spot_file}')
            continue
        spot = json.load(open(path))
        gs = GameState.from_dict(spot['game_state'])
        legal = spot['game_state']['legal_actions']
        # Restrict to actions the model outputs (0..4)
        legal_for_model = [a for a in legal if 0 <= a < 5]
        if not legal_for_model:
            legal_for_model = [a for a in legal if a < 6]

        # Build obs with NEW merged code path (53-dim, tier-aware)
        try:
            obs = builder.build(gs,
                                 hero_has_initiative_override=hero_init_override,
                                 hand_raise_count_override=raise_count_override)
        except Exception as e:
            print(f'\n{path.name}: obs build error: {e}')
            continue
        if obs is None:
            print(f'\n{path.name}: obs build returned None')
            continue
        # Trim equity_vs_odds (51st dim added by agent) if present
        # Our reference is 53-dim engine obs; agent adds +1 (EVO) → 54
        if len(obs) == 54:
            obs_53 = obs[:53]
        else:
            obs_53 = obs

        # Run both models
        with torch.no_grad():
            t = torch.from_numpy(obs_53.astype(np.float32)).unsqueeze(0)
            new_adv = new_model(t).numpy()[0]
            old_adv = old_model(t).numpy()[0]
        new_strat = _strategy_from_logits(new_adv, legal_for_model)
        old_strat = _strategy_from_logits(old_adv, legal_for_model)

        # Print spot summary
        print('\n' + '=' * 78)
        print(f'SPOT: {path.name}')
        print('=' * 78)
        hole = [str(c) for c in gs.community_cards] if False else \
                ['{}{}'.format(c.rank, c.suit)
                  for c in (gs.players[gs.my_seat].hole_cards or [])]
        board = ['{}{}'.format(c.rank, c.suit) for c in gs.community_cards]
        print(f'Hole: {"".join(hole)}  Board: {"".join(board)}  Street: {gs.street}')
        print(f'Pot: {gs.pot_total} BB  Call amount: {gs.call_amount} BB')
        print(f'Logged action: {spot["action_name"]} (idx {spot["action_idx"]})')
        # Logged strategy (from original 54-dim model run)
        log_strat = spot.get('strategy', [])
        if log_strat:
            print(f'Logged strategy (54-dim model at runtime):')
            for i, p in enumerate(log_strat[:5]):
                if p > 0.001:
                    print(f'  {ACTION_NAMES[i]:<10} {100*p:5.1f}%')

        # Print key obs values
        print(f'\nObs values (53-dim, NEW merged equity):')
        print(f'  obs[10] pot_odds:       {obs_53[10]:.3f}')
        print(f'  obs[18] hand_strength:  {obs_53[18]:.3f}  (now tier-aware)')
        print(f'  obs[19] made_hand_cat:  {obs_53[19]:.3f}')
        print(f'  obs[22] top_pair:       {obs_53[22]:.3f}')
        print(f'  obs[24] flush_texture:  {obs_53[24]:.3f}')
        print(f'  obs[25] straight_tex:   {obs_53[25]:.3f}')

        # The old obs[18] (equity-vs-random) from the log for comparison
        log_obs = spot.get('obs_vector', [])
        if len(log_obs) >= 19:
            print(f'\nLogged obs[18] (OLD equity-vs-random): {log_obs[18]:.3f}')
            print(f'New obs[18] (tier-aware merged):       {obs_53[18]:.3f}')
            print(f'Δ (new - old):  {obs_53[18] - log_obs[18]:+.3f}')

        # Strategies
        print(f'\nModel strategies on this obs:')
        print(f'  {"Action":<10} {"OLD iter_4000":>14}  {"NEW v55/4100":>14}  {"Δ":>8}')
        for a in legal_for_model:
            if a >= len(ACTION_NAMES):
                continue
            old_p = old_strat[a]
            new_p = new_strat[a]
            print(f'  {ACTION_NAMES[a]:<10} {100*old_p:>13.1f}%  {100*new_p:>13.1f}%  '
                  f'{100*(new_p - old_p):>+7.1f}%')


if __name__ == '__main__':
    main()
