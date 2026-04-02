"""Shared utilities for analysis scripts — parallel parsing + caching.

IMPORTANT: call parallel_parse() BEFORE loading torch/CUDA models.
mp.Pool with fork doesn't work after CUDA initialization.
"""
import hashlib
import multiprocessing as mp
import os
import pickle
from pathlib import Path

_CACHE_DIR = Path(__file__).parent / '.parse_cache'


def _parse_worker_showdown(filepath):
    """Parse a single HH file, return showdown hands only."""
    try:
        from src.hand_history.parser import parse_file
        hands = parse_file(str(filepath))
        return [h for h in hands if h.has_showdown and h.num_active_players == 6
                and abs(h.big_blind_dollars - 0.05) < 0.001]
    except Exception:
        return []


def _parse_worker_all(filepath):
    """Parse a single HH file, return ALL NL5 6-max hands."""
    try:
        from src.hand_history.parser import parse_file
        hands = parse_file(str(filepath))
        return [h for h in hands
                if abs(h.big_blind_dollars - 0.05) < 0.001
                and h.num_active_players >= 2]
    except Exception:
        return []


def _cache_key(hh_files, max_hands, showdown_only):
    """Generate cache key from file list + params."""
    file_sig = f'{len(hh_files)}|{str(hh_files[0]) if hh_files else ""}|{str(hh_files[-1]) if hh_files else ""}'
    key = f'{file_sig}|max={max_hands}|sd={showdown_only}'
    return hashlib.md5(key.encode()).hexdigest()[:12]


def parallel_parse(hh_files, max_hands=0, workers=0, showdown_only=True,
                   no_cache=False):
    """Parse HH files in parallel with disk caching.

    First call parses and caches. Subsequent calls load from cache instantly.
    Use no_cache=True to force re-parse.

    Args:
        showdown_only: True = only showdown hands, False = all hands
        no_cache: force re-parse, ignore cache
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_hash = _cache_key(hh_files, max_hands, showdown_only)
    label = 'showdown' if showdown_only else 'total'
    cache_file = _CACHE_DIR / f'parsed_{label}_{cache_hash}.pkl'

    # Try loading from cache
    if not no_cache and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                all_hands = pickle.load(f)
            mb = os.path.getsize(cache_file) / 1e6
            print(f'  Loaded {len(all_hands)} {label} hands from cache ({mb:.0f} MB)')
            return all_hands
        except Exception as e:
            print(f'  Cache load failed: {e}, re-parsing...')

    # Parse in parallel
    n_workers = workers if workers > 0 else min(mp.cpu_count(), 32)
    worker_fn = _parse_worker_showdown if showdown_only else _parse_worker_all
    all_hands = []

    # Limit files to parse: estimate ~50 hands/file for showdown, ~200 for all
    hands_per_file = 50 if showdown_only else 200
    if max_hands > 0:
        files_needed = min(len(hh_files), max(max_hands // hands_per_file + 100, 500))
        hh_files = hh_files[:files_needed]

    print(f'  Parsing {len(hh_files)} files with {n_workers} workers ({label})...')

    with mp.Pool(n_workers) as pool:
        for i, batch in enumerate(pool.imap_unordered(worker_fn, hh_files, chunksize=20)):
            all_hands.extend(batch)
            if (i + 1) % 500 == 0:
                print(f'    [{i+1}/{len(hh_files)}] {len(all_hands)} hands...',
                      flush=True)
            if max_hands > 0 and len(all_hands) >= max_hands:
                pool.terminate()
                break

    if max_hands > 0:
        all_hands = all_hands[:max_hands]

    print(f'  Parsed {len(all_hands)} {label} hands')

    # Cache only if reasonable size (< 50K hands), otherwise too slow to pickle
    if len(all_hands) <= 50000:
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(all_hands, f, protocol=pickle.HIGHEST_PROTOCOL)
            mb = os.path.getsize(cache_file) / 1e6
            print(f'  Cached to {cache_file.name} ({mb:.0f} MB)')
        except Exception as e:
            print(f'  Cache save failed: {e}')
    else:
        print(f'  Skipping cache ({len(all_hands)} hands too large to pickle)')

    return all_hands
