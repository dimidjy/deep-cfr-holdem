"""
Shared analysis database: parse HH files ONCE, cache result, share across analyses.

IMPORTANT: Call build_analysis_db() BEFORE any torch/CUDA import.
mp.Pool with fork deadlocks after CUDA initialization.

Usage:
    from scripts.analysis.analysis_db import build_analysis_db, AnalysisDB
    db = build_analysis_db('data/poker_archive/', max_hands=20000, workers=40)
"""

import hashlib
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# ── Project setup ───────────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.hand_history.parser import parse_file
from src.hand_history import ObservationBuilder, DecisionPoint
from src.hand_history.models import ParsedHand, dollars_to_chips, DOLLARS_TO_CHIPS

# ── Cache directory ─────────────────────────────────────────────────────────
_CACHE_DIR = _SCRIPT_DIR / '.parse_cache'

# ── Constants ───────────────────────────────────────────────────────────────
BIG_BLIND_DOLLARS = 0.05
NUM_PLAYERS = 6


# ── AnalysisDB dataclass ───────────────────────────────────────────────────

@dataclass
class AnalysisDecisionPoint:
    """Lightweight decision point for analysis (avoids pickling large obs arrays repeatedly)."""
    observation: np.ndarray
    legal_actions: List[int]
    actual_action: int
    street: str
    pot_dollars: float
    player_name: str
    player_idx: int
    hand_id: str
    hole_cards: Optional[List[str]]
    community_cards: List[str]
    position_label: str
    player_net_dollars: float
    is_last_action: bool


@dataclass
class AnalysisDB:
    """Cached analysis database containing all parsed data needed by analyzers."""
    # Decision points (flat list)
    decision_points: List[AnalysisDecisionPoint]

    # Raw ParsedHand objects for full_hand_replay and population_stats
    hands: List[ParsedHand]

    # Per-hand metadata
    hand_player_nets: Dict[str, Dict[str, float]]  # hand_id -> {player_name: net_dollars}

    # Player lists
    all_player_names: Set[str]

    # Stats
    total_hands_parsed: int = 0
    total_showdown_hands: int = 0
    total_decision_points: int = 0
    parse_time_seconds: float = 0.0

    # Include non-showdown hands too (for population stats)
    all_hands: Optional[List[ParsedHand]] = None


# ── Worker functions (module-level for multiprocessing pickling) ────────────

_PARSE_CFG: Dict = {}


def _parse_worker_init(num_players: int, use_behavioral_stats: bool):
    """Initialize per-worker state."""
    _PARSE_CFG['builder'] = ObservationBuilder(num_players=num_players)
    _PARSE_CFG['behavioral_stats'] = use_behavioral_stats
    _PARSE_CFG['num_players'] = num_players


def _compute_player_net(hand: ParsedHand) -> Dict[str, float]:
    """Compute net profit/loss for each active player in a hand (in dollars)."""
    player_invested = defaultdict(float)
    round_committed = defaultdict(float)
    current_street = 'preflop'

    for action in hand.actions:
        if action.street != current_street:
            current_street = action.street
            round_committed.clear()

        name = action.player_name
        if action.is_blind:
            player_invested[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'call':
            player_invested[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'bet':
            player_invested[name] += action.amount_dollars
            round_committed[name] += action.amount_dollars
        elif action.action_type == 'raise':
            to_amt = action.amount_dollars
            already = round_committed.get(name, 0)
            delta = to_amt - already
            if delta > 0:
                player_invested[name] += delta
            round_committed[name] = to_amt

    player_won = defaultdict(float)
    for w in hand.result.winners:
        player_won[w['name']] += w['amount_dollars']

    player_net = {}
    for p in hand.active_players:
        player_net[p.name] = player_won.get(p.name, 0.0) - player_invested.get(p.name, 0.0)
    return player_net


def _parse_single_file_showdown(file_path):
    """Parse one HH file, build decision points for showdown hands.

    Returns: (list_of_dicts, total_hands_in_file, showdown_hands_count)
    Each dict has: 'hand', 'dps', 'player_net', 'hand_id'
    """
    builder = _PARSE_CFG['builder']
    use_bh = _PARSE_CFG['behavioral_stats']
    num_players = _PARSE_CFG['num_players']

    file_results = []
    file_total = 0
    file_showdown = 0

    try:
        hands = parse_file(str(file_path))
    except Exception:
        return file_results, file_total, file_showdown

    for hand in hands:
        if abs(hand.big_blind_dollars - BIG_BLIND_DOLLARS) > 0.001:
            continue
        if hand.num_active_players != num_players:
            continue

        file_total += 1
        if not hand.has_showdown:
            continue

        file_showdown += 1
        player_net = _compute_player_net(hand)

        # Build decision points
        try:
            dps = builder.build_decision_points(
                hand,
                allow_unknown_cards=False,
                behavioral_stats=use_bh,
            )
        except Exception:
            dps = []

        # Convert DecisionPoints to lightweight AnalysisDecisionPoints
        adps = []
        for dp in dps:
            adps.append(AnalysisDecisionPoint(
                observation=dp.observation,
                legal_actions=dp.legal_actions,
                actual_action=dp.actual_action,
                street=dp.street,
                pot_dollars=dp.pot_dollars,
                player_name=dp.player_name,
                player_idx=dp.player_idx,
                hand_id=dp.hand_id,
                hole_cards=dp.hole_cards,
                community_cards=dp.community_cards,
                position_label=dp.position_label,
                player_net_dollars=player_net.get(dp.player_name, 0.0),
                is_last_action=dp.is_last_action,
            ))

        file_results.append({
            'hand': hand,
            'dps': adps,
            'player_net': player_net,
            'hand_id': hand.hand_id,
        })

    return file_results, file_total, file_showdown


def _parse_single_file_all(file_path):
    """Parse one HH file, return ALL NL5 6-max hands (for population stats)."""
    try:
        hands = parse_file(str(file_path))
        return [h for h in hands
                if abs(h.big_blind_dollars - BIG_BLIND_DOLLARS) < 0.001
                and h.num_active_players >= 2]
    except Exception:
        return []


# ── Cache key ───────────────────────────────────────────────────────────────

def _cache_key(hh_files: list, max_hands: int, include_all_hands: bool) -> str:
    """Generate stable cache key from file list and params."""
    file_sig = '|'.join([
        str(len(hh_files)),
        str(hh_files[0]) if hh_files else '',
        str(hh_files[-1]) if hh_files else '',
        str(hh_files[len(hh_files)//2]) if len(hh_files) > 2 else '',
    ])
    key = f'{file_sig}|max={max_hands}|all={include_all_hands}'
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ── Public API ──────────────────────────────────────────────────────────────

def build_analysis_db(
    hh_dir: str,
    max_hands: int = 0,
    workers: int = 0,
    no_cache: bool = False,
    include_all_hands: bool = False,
    num_players: int = NUM_PLAYERS,
) -> AnalysisDB:
    """Parse HH files and build the shared AnalysisDB.

    Args:
        hh_dir: Directory containing PokerStars .txt HH files.
        max_hands: Maximum showdown hands to process (0 = all).
        workers: Number of parser workers (0 = auto).
        no_cache: Force re-parse, ignore disk cache.
        include_all_hands: Also parse non-showdown hands (for population_stats).
        num_players: Expected table size (default 6).

    Returns:
        AnalysisDB with all parsed data.
    """
    _CACHE_DIR.mkdir(exist_ok=True)

    hh_path = Path(hh_dir)
    if not hh_path.is_absolute():
        hh_path = _PROJECT_ROOT / hh_path
    hh_files = sorted(hh_path.rglob('*.txt'))
    print(f'[AnalysisDB] Found {len(hh_files)} HH files in {hh_path}')

    if not hh_files:
        print('[AnalysisDB] ERROR: No .txt files found.')
        return AnalysisDB(
            decision_points=[], hands=[], hand_player_nets={},
            all_player_names=set(),
        )

    # Check cache
    cache_hash = _cache_key(hh_files, max_hands, include_all_hands)
    cache_file = _CACHE_DIR / f'analysis_db_{cache_hash}.pkl'

    if not no_cache and cache_file.exists():
        try:
            print(f'[AnalysisDB] Loading from cache: {cache_file.name}')
            t0 = time.time()
            with open(cache_file, 'rb') as f:
                db = pickle.load(f)
            mb = os.path.getsize(cache_file) / 1e6
            print(f'[AnalysisDB] Loaded {db.total_showdown_hands} showdown hands, '
                  f'{db.total_decision_points} DPs from cache '
                  f'({mb:.0f} MB, {time.time()-t0:.1f}s)')
            return db
        except Exception as e:
            print(f'[AnalysisDB] Cache load failed: {e}, re-parsing...')

    # Parse showdown hands with decision points
    n_workers = workers if workers > 0 else min(mp.cpu_count(), 32)
    print(f'[AnalysisDB] Parsing showdown hands with {n_workers} workers...')
    t0 = time.time()

    # Estimate files needed
    if max_hands > 0:
        files_needed = min(len(hh_files), max(max_hands // 10 + 200, 500))
        parse_files = hh_files[:files_needed]
    else:
        parse_files = hh_files

    # Limit files: ~50 showdown hands per file on average
    if max_hands > 0:
        files_needed = min(len(parse_files), max_hands // 50 + 100)
        parse_files = parse_files[:files_needed]
        print(f'[AnalysisDB] Limited to {len(parse_files)} files '
              f'(est. {files_needed * 30} hands)', flush=True)

    all_results = []
    total_file_hands = 0
    total_showdown = 0

    pool = mp.Pool(
        processes=n_workers,
        initializer=_parse_worker_init,
        initargs=(num_players, True),
    )
    try:
        for batch in pool.imap_unordered(
            _parse_single_file_showdown, parse_files, chunksize=1
        ):
            file_results, file_total, file_sd = batch
            all_results.extend(file_results)
            total_file_hands += file_total
            total_showdown += file_sd

            if total_showdown % 5000 < 50:
                print(f'  {total_showdown} showdown hands '
                      f'({total_file_hands} total)...', flush=True)

            if max_hands > 0 and total_showdown >= max_hands:
                break
    finally:
        pool.close()
        pool.join()

    print(f'  Pool done, collecting results...', flush=True)

    if max_hands > 0:
        all_results = all_results[:max_hands]

    parse_time = time.time() - t0
    print(f'[AnalysisDB] Parsed {len(all_results)} showdown hands in {parse_time:.1f}s',
          flush=True)

    # Build AnalysisDB
    print(f'[AnalysisDB] Building database from {len(all_results)} hands...', flush=True)
    decision_points = []
    hands = []
    hand_player_nets = {}
    all_player_names = set()

    for r in all_results:
        hand = r['hand']
        hands.append(hand)
        hand_player_nets[r['hand_id']] = r['player_net']
        decision_points.extend(r['dps'])
        for p in hand.active_players:
            all_player_names.add(p.name)

    # Use showdown hands for all analyses.
    # Population stats can work with showdown hands only (limited but usable).
    all_hands_list = hands if include_all_hands else None

    db = AnalysisDB(
        decision_points=decision_points,
        hands=hands,
        hand_player_nets=hand_player_nets,
        all_player_names=all_player_names,
        total_hands_parsed=total_file_hands,
        total_showdown_hands=len(hands),
        total_decision_points=len(decision_points),
        parse_time_seconds=parse_time,
        all_hands=all_hands_list,
    )

    # Cache to disk (skip if too large — pickle is slow for big objects)
    if len(hands) <= 10000:
        try:
            print(f'[AnalysisDB] Caching {len(hands)} hands...', flush=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
            mb = os.path.getsize(cache_file) / 1e6
            print(f'[AnalysisDB] Cached to {cache_file.name} ({mb:.0f} MB)')
        except Exception as e:
            print(f'[AnalysisDB] Cache save failed: {e}')
    else:
        print(f'[AnalysisDB] Skipping cache ({len(hands)} hands — too large to pickle)')

    return db


def load_analysis_db(path: str) -> AnalysisDB:
    """Load a previously saved AnalysisDB from a pickle file."""
    with open(path, 'rb') as f:
        db = pickle.load(f)
    if not isinstance(db, AnalysisDB):
        raise TypeError(f'Expected AnalysisDB, got {type(db).__name__}')
    return db


def load_player_names(filepath: str) -> Set[str]:
    """Load player names from a text file (one per line, # comments)."""
    names = set()
    full_path = filepath if os.path.isabs(filepath) else str(_PROJECT_ROOT / filepath)
    try:
        with open(full_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    names.add(line)
    except FileNotFoundError:
        print(f'[AnalysisDB] Warning: player file not found: {full_path}')
    return names


def compute_player_winrates(db: AnalysisDB) -> Dict[str, float]:
    """Compute per-player BB/100 win rates from the parsed data."""
    player_hands = defaultdict(int)
    player_bb_net = defaultdict(float)

    for hand in db.hands:
        hand_nets = db.hand_player_nets.get(hand.hand_id, {})
        bb = hand.big_blind_dollars
        for p in hand.active_players:
            player_hands[p.name] += 1
            net_bb = hand_nets.get(p.name, 0.0) / bb if bb > 0 else 0.0
            player_bb_net[p.name] += net_bb

    player_bb100 = {}
    for name in player_hands:
        n = player_hands[name]
        if n >= 100:
            player_bb100[name] = (player_bb_net[name] / n) * 100.0
    return player_bb100
