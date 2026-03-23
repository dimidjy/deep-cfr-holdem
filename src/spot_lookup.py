"""
Spot lookup engine — find similar historical spots and return action outcomes.

Given a current game observation, finds the N most similar decision points
from the database and returns aggregated statistics per action.

Usage:
    from src.spot_lookup import SpotLookup

    lookup = SpotLookup('data/spot_db.npz')
    results = lookup.query(observation, street='flop', position='CO', k=200)
    print(results)
    # {
    #   'fold':  {'count': 12, 'avg_bb': -1.2, 'win_rate': 0.0},
    #   'call':  {'count': 95, 'avg_bb': 0.8, 'win_rate': 0.58},
    #   'raise': {'count': 83, 'avg_bb': 2.1, 'win_rate': 0.64},
    #   'all_in':{'count': 10, 'avg_bb': 5.3, 'win_rate': 0.70},
    #   'total_matches': 200,
    #   'avg_distance': 0.15,
    # }
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from scipy.spatial import KDTree


ACTION_NAMES = {0: 'fold', 1: 'check_call', 2: 'raise_half', 3: 'raise_pot', 4: 'all_in'}
STREET_MAP = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
POS_MAP = {'UTG': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}


class SpotLookup:
    """Fast nearest-neighbor spot lookup using KDTree per street."""

    def __init__(self, db_path: str):
        data = np.load(db_path)
        self.observations = data['observations']   # (N, obs_dim)
        self.actions = data['actions']              # (N,)
        self.net_bb = data['net_bb']                # (N,)
        self.streets = data['streets']              # (N,)
        self.positions = data['positions']          # (N,)

        # Build per-street KDTree for faster search
        self.trees: Dict[int, KDTree] = {}
        self.indices: Dict[int, np.ndarray] = {}

        for street_idx in range(4):
            mask = self.streets == street_idx
            idx = np.where(mask)[0]
            if len(idx) > 0:
                self.trees[street_idx] = KDTree(self.observations[idx])
                self.indices[street_idx] = idx

        total = len(self.observations)
        print(f'SpotLookup loaded: {total} spots')
        for s_name, s_idx in STREET_MAP.items():
            count = len(self.indices.get(s_idx, []))
            print(f'  {s_name}: {count}')

    def query(self, observation: np.ndarray,
              street: str = 'flop',
              position: Optional[str] = None,
              k: int = 200,
              position_filter: bool = False) -> Dict:
        """
        Find k most similar spots and return action outcome statistics.

        Args:
            observation: Feature vector (same dim as training obs)
            street: 'preflop', 'flop', 'turn', 'river'
            position: Optional position filter ('UTG','MP','CO','BTN','SB','BB')
            k: Number of nearest neighbors
            position_filter: If True, only return spots from same position

        Returns:
            Dict with per-action stats and metadata
        """
        street_idx = STREET_MAP.get(street, 1)

        if street_idx not in self.trees:
            return {'error': f'No spots for street {street}', 'total_matches': 0}

        tree = self.trees[street_idx]
        idx_map = self.indices[street_idx]

        # Query more than k if we'll filter by position
        query_k = min(k * 5 if position_filter and position else k, len(idx_map))
        obs = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        distances, local_indices = tree.query(obs, k=query_k)
        distances = distances[0]
        local_indices = local_indices[0]

        # Map back to global indices
        global_indices = idx_map[local_indices]

        # Optional position filter
        if position_filter and position:
            pos_idx = POS_MAP.get(position, -1)
            if pos_idx >= 0:
                pos_mask = self.positions[global_indices] == pos_idx
                global_indices = global_indices[pos_mask]
                distances = distances[pos_mask]

        # Trim to k
        global_indices = global_indices[:k]
        distances = distances[:k]

        if len(global_indices) == 0:
            return {'total_matches': 0, 'avg_distance': float('inf')}

        # Aggregate per action
        matched_actions = self.actions[global_indices]
        matched_net = self.net_bb[global_indices]

        result = {}
        for action_id, action_name in ACTION_NAMES.items():
            mask = matched_actions == action_id
            count = int(mask.sum())
            if count == 0:
                result[action_name] = {'count': 0, 'avg_bb': 0.0, 'win_rate': 0.0,
                                       'median_bb': 0.0, 'std_bb': 0.0}
                continue

            net_vals = matched_net[mask]
            win_count = int((net_vals > 0).sum())
            result[action_name] = {
                'count': count,
                'avg_bb': float(net_vals.mean()),
                'median_bb': float(np.median(net_vals)),
                'std_bb': float(net_vals.std()),
                'win_rate': win_count / count,
            }

        # Group aggressive actions for summary
        raise_actions = matched_actions >= 2
        call_actions = matched_actions == 1
        fold_actions = matched_actions == 0

        summary = {}
        for label, mask in [('fold', fold_actions), ('passive', call_actions),
                            ('aggressive', raise_actions)]:
            count = int(mask.sum())
            if count > 0:
                net_vals = matched_net[mask]
                summary[label] = {
                    'count': count,
                    'pct': count / len(global_indices) * 100,
                    'avg_bb': float(net_vals.mean()),
                    'win_rate': float((net_vals > 0).sum()) / count,
                }

        result['summary'] = summary
        result['total_matches'] = len(global_indices)
        result['avg_distance'] = float(distances.mean())
        result['max_distance'] = float(distances.max())

        return result

    def query_pretty(self, observation: np.ndarray,
                     street: str = 'flop',
                     position: Optional[str] = None,
                     k: int = 200) -> str:
        """Human-readable query result."""
        r = self.query(observation, street, position, k)

        if r['total_matches'] == 0:
            return 'No similar spots found.'

        lines = [
            f'Found {r["total_matches"]} similar spots (avg dist: {r["avg_distance"]:.3f})',
            '',
            f'{"Action":<14} {"Count":>6} {"Avg BB":>8} {"Med BB":>8} {"Win%":>6}',
            '-' * 46,
        ]

        for action_name in ['fold', 'check_call', 'raise_half', 'raise_pot', 'all_in']:
            s = r.get(action_name, {})
            if s.get('count', 0) == 0:
                continue
            lines.append(
                f'{action_name:<14} {s["count"]:>6} {s["avg_bb"]:>+8.2f} '
                f'{s["median_bb"]:>+8.2f} {s["win_rate"]*100:>5.1f}%'
            )

        if 'summary' in r:
            lines.append('')
            lines.append('Summary:')
            for label in ['fold', 'passive', 'aggressive']:
                s = r['summary'].get(label, {})
                if s.get('count', 0) == 0:
                    continue
                lines.append(
                    f'  {label:<12} {s["count"]:>5} ({s["pct"]:>5.1f}%)  '
                    f'avg {s["avg_bb"]:>+.2f} BB  win {s["win_rate"]*100:.0f}%'
                )

        return '\n'.join(lines)
