#!/usr/bin/env python3
"""
Automated training controller for Deep CFR 6-player poker.

What it does:
  - Runs train_gpu.py as a subprocess
  - Reads validation output every checkpoint_interval iterations
  - If checkpoint is GOOD  → adds it to the frozen opponent pool
  - If checkpoint is BAD for N consecutive checks → stops and restarts
    from the last good checkpoint with the updated pool
  - Persists pool state to disk so restarts survive controller crashes

Usage:
  python3 scripts/train_controller.py \
      --initial-model models/6player_transferred.pt \
      --device cuda --workers 60

Good checkpoint criteria (configurable):
  - trash_offsuit fold rate >= TRASH_FOLD_MIN  (default 20%)
  - OR spots_passing >= SPOTS_MIN              (default 3 out of 6)
"""

import sys
import os
import json
import re
import time
import signal
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Quality thresholds
# ---------------------------------------------------------------------------
TRASH_FOLD_MIN      = 0.20   # trash_offsuit must fold >=20% for a "good" checkpoint
SPOTS_MIN           = 3      # OR at least N/6 spots must pass
CONSECUTIVE_BAD_MAX = 3      # restart after this many consecutive bad validations
MAX_POOL_SIZE       = 8      # keep at most this many checkpoints in the pool
                             # (always preserves slot 0 = initial model)

POOL_MANIFEST_PATH = "models/pool_manifest.json"
LOG_PATH           = "data/logs/controller.log"


# ---------------------------------------------------------------------------
# Pool manifest helpers
# ---------------------------------------------------------------------------

def load_manifest():
    p = Path(POOL_MANIFEST_PATH)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"pool": [], "best_checkpoint": None}


def save_manifest(m):
    Path(POOL_MANIFEST_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(POOL_MANIFEST_PATH, "w") as f:
        json.dump(m, f, indent=2)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_iter_line(line):
    """Return (iter_num, total) from 'Iter  300/2000  ...' line, or (None, None)."""
    m = re.search(r'Iter\s+(\d+)/(\d+)', line)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def parse_validation_line(line):
    """
    Parse '  [spots 2/6] AA:fold=1%,...  trash_offsuit:fold=6%,...' line.
    Returns dict or None.
    """
    sm = re.search(r'\[spots\s+(\d+)/(\d+)\]', line)
    if not sm:
        return None

    hands = {}
    for m in re.finditer(r'([\w]+):fold=(\d+)%', line):
        hands[m.group(1)] = int(m.group(2)) / 100.0

    return {
        'spots_passing': int(sm.group(1)),
        'spots_total':   int(sm.group(2)),
        'trash_fold':    hands.get('trash_offsuit', 0.0),
        '72o_fold':      hands.get('72o', 0.0),
        'aa_fold':       hands.get('AA', 1.0),
        'hands':         hands,
    }


def is_good(metrics):
    if metrics is None:
        return False
    return (metrics['trash_fold'] >= TRASH_FOLD_MIN or
            metrics['spots_passing'] >= SPOTS_MIN)


# ---------------------------------------------------------------------------
# Single training run (subprocess)
# ---------------------------------------------------------------------------

def run_training_session(resume_path, pool_paths, device, workers, models_dir):
    """
    Launch train_gpu.py, monitor output, return when done or when degradation
    is detected.

    Returns:
        last_iter   – last iteration number seen
        manifest    – updated manifest dict
        reason      – 'degraded' | 'completed' | 'interrupted'
    """
    manifest = load_manifest()

    script = Path(__file__).parent / "train_gpu.py"
    cmd = [sys.executable, str(script)]
    if resume_path:
        cmd += ["--resume", str(resume_path)]
    if pool_paths:
        cmd += ["--frozen-opponents"] + [str(p) for p in pool_paths]
    if device:
        cmd += ["--device", device]
    if workers:
        cmd += ["--workers", str(workers)]

    print(f"\n{'='*60}")
    print(f"[CTRL] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[CTRL] resume        : {resume_path}")
    print(f"[CTRL] pool ({len(pool_paths):2d})    : {[str(p) for p in pool_paths]}")
    print(f"[CTRL] command       : {' '.join(cmd)}")
    print('='*60)

    Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    consecutive_bad = 0
    last_iter = 0
    reason = "completed"

    with open(LOG_PATH, "a") as log_f:
        log_f.write(f"\n\n--- session {datetime.now()} ---\n")
        log_f.write(f"cmd: {' '.join(cmd)}\n")

        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log_f.write(line)
                log_f.flush()

                # Track current iteration
                it, _ = parse_iter_line(line)
                if it:
                    last_iter = it

                # Process validation checkpoint lines
                if "[spots" not in line:
                    continue

                metrics = parse_validation_line(line)
                if metrics is None:
                    continue

                ckpt_path = Path(models_dir) / f"checkpoint_iter_{last_iter}.pt"

                print(f"\n[CTRL] Validation @ iter {last_iter}:"
                      f"  spots={metrics['spots_passing']}/{metrics['spots_total']}"
                      f"  trash_fold={metrics['trash_fold']:.0%}"
                      f"  72o_fold={metrics['72o_fold']:.0%}")

                if is_good(metrics):
                    consecutive_bad = 0
                    if ckpt_path.exists():
                        ckpt_str = str(ckpt_path)
                        if ckpt_str not in manifest["pool"]:
                            manifest["pool"].append(ckpt_str)
                            # Trim pool: keep slot-0 (initial model) + newest ones
                            if len(manifest["pool"]) > MAX_POOL_SIZE:
                                manifest["pool"] = (manifest["pool"][:1] +
                                                    manifest["pool"][-(MAX_POOL_SIZE-1):])
                            manifest["best_checkpoint"] = ckpt_str
                            manifest["validated_checkpoint"] = ckpt_str
                            save_manifest(manifest)
                            print(f"[CTRL] GOOD → added to pool  (pool size={len(manifest['pool'])})")
                        else:
                            print(f"[CTRL] GOOD (already in pool)")
                    else:
                        print(f"[CTRL] GOOD but checkpoint file not found: {ckpt_path}")
                else:
                    consecutive_bad += 1
                    print(f"[CTRL] BAD  ({consecutive_bad}/{CONSECUTIVE_BAD_MAX})")
                    if consecutive_bad >= CONSECUTIVE_BAD_MAX:
                        print(f"[CTRL] Too many bad validations — stopping this run.")
                        proc.terminate()
                        reason = "degraded"
                        break

        except KeyboardInterrupt:
            print("\n[CTRL] Interrupted.")
            proc.terminate()
            reason = "interrupted"

    proc.wait()
    return last_iter, manifest, reason


# ---------------------------------------------------------------------------
# Controller main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Deep CFR training controller — auto pool management & restart"
    )
    parser.add_argument("--initial-model", type=str, required=True,
                        help="Starting checkpoint (e.g. models/6player_transferred.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda | cpu  (auto-detected if omitted)")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU worker count")
    parser.add_argument("--max-restarts", type=int, default=20,
                        help="Maximum number of restarts on degradation (default: 20)")
    parser.add_argument("--models-dir", type=str, default="models",
                        help="Directory where checkpoints are saved")
    parser.add_argument("--reset-pool", action="store_true",
                        help="Ignore existing pool manifest and start fresh")
    args = parser.parse_args()

    # Resolve paths relative to repo root (parent of scripts/)
    repo_root = Path(__file__).parent.parent
    os.chdir(repo_root)

    initial_model = Path(args.initial_model)
    if not initial_model.exists():
        print(f"[CTRL] ERROR: initial model not found: {initial_model}")
        sys.exit(1)

    # Load or initialise manifest
    manifest = {} if args.reset_pool else load_manifest()
    if not manifest.get("pool"):
        manifest["pool"] = [str(initial_model)]
        manifest["best_checkpoint"] = None   # None = fresh start, no --resume
        manifest["validated_checkpoint"] = None
        save_manifest(manifest)

    print(f"[CTRL] Starting controller")
    print(f"[CTRL] Initial model : {initial_model}")
    print(f"[CTRL] Pool          : {manifest['pool']}")
    print(f"[CTRL] Thresholds    : trash_fold>={TRASH_FOLD_MIN:.0%}  "
          f"OR spots>={SPOTS_MIN}/6")
    print(f"[CTRL] Max restarts  : {args.max_restarts}")

    for run_num in range(1, args.max_restarts + 1):
        print(f"\n[CTRL] ===== RUN {run_num}/{args.max_restarts} =====")

        # Use last validated checkpoint as resume (None = fresh network weights on first run)
        resume_path = manifest.get("validated_checkpoint")
        pool_paths  = manifest["pool"]

        last_iter, manifest, reason = run_training_session(
            resume_path=resume_path,
            pool_paths=pool_paths,
            device=args.device,
            workers=args.workers,
            models_dir=args.models_dir,
        )

        print(f"\n[CTRL] Run {run_num} ended: reason={reason}  last_iter={last_iter}")
        print(f"[CTRL] Pool size={len(manifest['pool'])}  best={manifest.get('best_checkpoint')}")

        if reason == "interrupted":
            print("[CTRL] Stopped by user.")
            break

        if reason == "completed":
            print("[CTRL] Training completed normally.")
            break

        # reason == "degraded" → restart
        if run_num == args.max_restarts:
            print("[CTRL] Reached max restarts — giving up.")
            break

        wait = 5
        print(f"[CTRL] Restarting in {wait}s with updated pool...")
        time.sleep(wait)

    print("\n[CTRL] Controller done.")


if __name__ == "__main__":
    main()
