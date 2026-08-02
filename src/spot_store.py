"""Persistent, editable spot collection used as a model regression suite.

`src/spot_library.py` generates spots parametrically — useful for sweeping,
but there is nowhere to record *what the right answer is*. This store keeps a
curated set where each spot carries a `right_action` you decide, so a model
can be scored against your own judgement rather than against a threshold.

Backed by SQLite at `data/spot_library.sqlite`. Rows are self-contained: the
cards, board, and sizing live on the row rather than referencing a generator,
so editing one spot can never shift another and a saved suite keeps meaning
after the generator changes.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DB = 'data/spot_library.sqlite'

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT 'custom',
    street        TEXT NOT NULL DEFAULT 'flop',
    position      TEXT NOT NULL DEFAULT 'IP',
    num_players   INTEGER NOT NULL DEFAULT 2,
    facing        TEXT NOT NULL DEFAULT 'checked_to',
    hole          TEXT NOT NULL DEFAULT '[]',      -- json list
    board         TEXT NOT NULL DEFAULT '[]',      -- json list
    pot_bb        REAL NOT NULL DEFAULT 7.0,
    call_bb       REAL NOT NULL DEFAULT 0.0,
    stack_bb      REAL NOT NULL DEFAULT 100.0,
    right_action  INTEGER,                          -- NULL = not yet labelled
    notes         TEXT NOT NULL DEFAULT '',
    tags          TEXT NOT NULL DEFAULT '',
    created_at    TEXT NOT NULL DEFAULT '',
    updated_at    TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_spots_cat ON spots(category);
"""

_FIELDS = ['label', 'category', 'street', 'position', 'num_players', 'facing',
           'hole', 'board', 'pot_bb', 'call_bb', 'stack_bb', 'right_action',
           'notes', 'tags']
_JSON_FIELDS = {'hole', 'board'}


def _now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S')


class SpotStore:
    def __init__(self, db_path: str = DEFAULT_DB) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(_SCHEMA)
        self.db.commit()

    # ── read ───────────────────────────────────────────────────────────────
    @staticmethod
    def _row(r: sqlite3.Row) -> Dict[str, Any]:
        d = dict(r)
        for f in _JSON_FIELDS:
            try:
                d[f] = json.loads(d.get(f) or '[]')
            except Exception:
                d[f] = []
        return d

    def list(self, category: str = '', labelled_only: bool = False
             ) -> List[Dict[str, Any]]:
        sql = 'SELECT * FROM spots'
        where, params = [], []
        if category:
            where.append('category = ?')
            params.append(category)
        if labelled_only:
            where.append('right_action IS NOT NULL')
        if where:
            sql += ' WHERE ' + ' AND '.join(where)
        sql += ' ORDER BY category, street, id'
        return [self._row(r) for r in self.db.execute(sql, params)]

    def get(self, spot_id: int) -> Optional[Dict[str, Any]]:
        r = self.db.execute('SELECT * FROM spots WHERE id = ?',
                            (spot_id,)).fetchone()
        return self._row(r) if r else None

    def categories(self) -> List[str]:
        return [r[0] for r in self.db.execute(
            'SELECT DISTINCT category FROM spots ORDER BY category')]

    def stats(self) -> Dict[str, int]:
        total = self.db.execute('SELECT COUNT(*) FROM spots').fetchone()[0]
        labelled = self.db.execute(
            'SELECT COUNT(*) FROM spots WHERE right_action IS NOT NULL'
        ).fetchone()[0]
        return {'total': total, 'labelled': labelled,
                'unlabelled': total - labelled}

    # ── write ──────────────────────────────────────────────────────────────
    @staticmethod
    def _encode(data: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for f in _FIELDS:
            if f not in data:
                continue
            v = data[f]
            out[f] = json.dumps(v) if f in _JSON_FIELDS else v
        return out

    def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        vals = self._encode(data)
        vals.setdefault('created_at', _now())
        vals['updated_at'] = _now()
        cols = ', '.join(vals)
        marks = ', '.join('?' * len(vals))
        cur = self.db.execute(
            f'INSERT INTO spots ({cols}) VALUES ({marks})', list(vals.values()))
        self.db.commit()
        return self.get(cur.lastrowid)

    def update(self, spot_id: int, data: Dict[str, Any]) -> Optional[Dict]:
        vals = self._encode(data)
        if not vals:
            return self.get(spot_id)
        vals['updated_at'] = _now()
        sets = ', '.join(f'{k} = ?' for k in vals)
        self.db.execute(f'UPDATE spots SET {sets} WHERE id = ?',
                        list(vals.values()) + [spot_id])
        self.db.commit()
        return self.get(spot_id)

    def delete(self, spot_id: int) -> bool:
        cur = self.db.execute('DELETE FROM spots WHERE id = ?', (spot_id,))
        self.db.commit()
        return cur.rowcount > 0

    def delete_all(self) -> int:
        cur = self.db.execute('DELETE FROM spots')
        self.db.commit()
        return cur.rowcount

    # ── bulk seeding ───────────────────────────────────────────────────────
    def seed_from_generator(self, **kwargs) -> int:
        """Add every spot the parametric generator produces for these filters.

        Right actions are left NULL — the point is that you label them.
        Duplicates (same cards + same situation) are skipped so re-seeding
        after widening a filter only adds what is new.
        """
        from src.spot_library import generate

        existing = {
            (r['category'], r['street'], r['position'], r['num_players'],
             r['facing'], tuple(r['hole']), tuple(r['board']))
            for r in self.list()
        }
        added = 0
        for s in generate(**kwargs):
            key = (s.category, s.street, s.position, s.num_players, s.facing,
                   tuple(s.hole), tuple(s.board))
            if key in existing:
                continue
            self.create({
                'label': f'{s.category_label} · {s.street} · {s.position} · '
                         f'{s.num_players}p · {s.facing_label}',
                'category': s.category, 'street': s.street,
                'position': s.position, 'num_players': s.num_players,
                'facing': s.facing, 'hole': s.hole, 'board': s.board,
                'pot_bb': s.pot_bb, 'call_bb': s.call_bb,
                'stack_bb': s.stack_bb, 'right_action': None,
                'tags': 'generated',
            })
            existing.add(key)
            added += 1
        return added


_SHARED: Optional[SpotStore] = None


def get_store(db_path: str = DEFAULT_DB) -> SpotStore:
    global _SHARED
    if _SHARED is None:
        _SHARED = SpotStore(db_path)
    return _SHARED


def row_to_spot(row: Dict[str, Any]):
    """Turn a stored row into a `spot_library.Spot` the evaluator can run."""
    from src.spot_library import CATEGORY_BY_KEY, FACING, Spot

    cat = CATEGORY_BY_KEY.get(row.get('category') or '', None)
    face = next((f for f in FACING if f['key'] == row.get('facing')), None)
    return Spot(
        id=str(row['id']),
        category=row.get('category') or 'custom',
        category_label=(cat['label'] if cat else (row.get('label') or 'Custom')),
        group=(cat['group'] if cat else 'custom'),
        rank=(cat['rank'] if cat else 99),
        street=row.get('street') or 'flop',
        position=row.get('position') or 'IP',
        num_players=int(row.get('num_players') or 2),
        facing=row.get('facing') or 'checked_to',
        facing_label=(face['label'] if face else (
            f"vs {row.get('call_bb')}bb" if row.get('call_bb') else 'Checked to')),
        hole=list(row.get('hole') or []),
        board=list(row.get('board') or []),
        pot_bb=float(row.get('pot_bb') or 7.0),
        call_bb=float(row.get('call_bb') or 0.0),
        stack_bb=float(row.get('stack_bb') or 100.0),
    )
