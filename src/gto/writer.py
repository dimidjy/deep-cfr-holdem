"""Write gto_* fields into a spot JSON, preserving every other key.

Mirrors the annotation-preservation contract in
`scripts/rebuild_spot_obs.py`: load the existing JSON, overwrite
only the gto_* keys this module owns, copy everything else (annotations like
`right_action` / `deleted` / `claude_*`, the obs, the game_state) forward
verbatim, then atomically replace the file.
"""
import json
import os
import tempfile
from datetime import datetime
from typing import Dict

# Every key this module owns. `write_gto_into_spot` clears all of these before
# applying the new fields, so a re-solve never leaves stale keys behind (e.g.
# a previous `gto_error` lingering after a later successful solve).
_GTO_KEYS = {
    'gto_strategy', 'gto_action', 'gto_freq_played', 'gto_node_actions',
    'gto_action_map', 'gto_spot_key', 'gto_range_source', 'gto_node',
    'gto_nav_type', 'gto_board', 'gto_pot_chips', 'gto_eff_stack_chips',
    'gto_solver_version', 'gto_solved_at', 'gto_error', 'gto_bluff_factor',
}


def write_gto_into_spot(path: str, fields: Dict) -> None:
    """Merge `fields` (gto_* only) into the spot JSON at `path` atomically."""
    with open(path) as f:
        data = json.load(f)
    for k in _GTO_KEYS:
        data.pop(k, None)
    for k, v in fields.items():
        data[k] = v
    data['gto_solved_at'] = datetime.now().isoformat(timespec='seconds')
    _atomic_dump(path, data)


def _atomic_dump(path: str, data: Dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=1, default=str)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
