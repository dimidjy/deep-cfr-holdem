"""Job manager: runs one project script at a time and streams its output.

Only ONE job may run at a time. That is a hard project rule, not a UI
convenience — training and validation scripts all compete for the same GPU
and CPU cores, and running them concurrently corrupts timings and starves
the trainer (see CLAUDE.md). `scripts/tournament.py` is the documented
exception: it parallelises internally via --workers, so it still counts as
a single job here.

Each run gets a directory under webui/runs/<job_id>/ holding the generated
config, the full log, and a metadata JSON, so a run can be audited or
reproduced later.
"""
from __future__ import annotations

import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = Path(__file__).resolve().parent / 'runs'
RUNS_DIR.mkdir(exist_ok=True)

MAX_LIVE_LINES = 4000       # ring buffer kept in memory for late-joining clients


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing — these patterns match what the scripts actually print.
# ─────────────────────────────────────────────────────────────────────────────

# Iter 1234/7000  buf=1234567  adv=0.0123↓  strat=0.0456↑  lr=0.00030  trav=12.3s  train=4.5s
_RE_ITER = re.compile(
    r'Iter\s+(?P<iter>\d+)/(?P<total>\d+)\s+'
    r'buf=\s*(?P<buf>\d+)\s+'
    r'adv=(?P<adv>[\d.]+)\D*?\s+'
    r'strat=(?P<strat>[\d.]+)\D*?\s+'
    r'lr=(?P<lr>[\d.]+)\s+'
    r'trav=(?P<trav>[\d.]+)s\s+'
    r'train=(?P<train>[\d.]+)s'
)
# [self-play 5000h] AF=2.45 | Fold=30% Chk=25% Call=20% Bet=35%
_RE_SELFPLAY = re.compile(
    r'\[self-play\s+(?P<hands>\d+)h\]\s+AF=(?P<af>[\d.]+)\s*\|\s*'
    r'Fold=(?P<fold>[\d.]+)%\s+Chk=(?P<chk>[\d.]+)%\s+'
    r'Call=(?P<call>[\d.]+)%\s+Bet=(?P<bet>[\d.]+)%'
)
# [spots 3301] score=3150.5/3301 (95.4%)  exact=3000  partial=301
_RE_SPOTS = re.compile(
    r'\[spots\s+(?P<n>\d+)\]\s+score=(?P<score>[\d.]+)/\d+\s+'
    r'\((?P<pct>[\d.]+)%\)\s+exact=(?P<exact>\d+)\s+partial=(?P<partial>\d+)'
)
_RE_FILLING = re.compile(r'FILLING BUFFER\s+(?P<have>\d+)/(?P<need>\d+)')
_RE_PHASE = re.compile(r'\bP(?P<n>[123]):(?P<name>[\w-]+)')
_RE_CKPT = re.compile(r'checkpoint_iter_(\d+)\.pt')


@dataclass
class Job:
    id: str
    kind: str                       # 'train' | 'self_play' | ...
    label: str
    cmd: List[str]
    status: str = 'running'         # running | finished | failed | stopped
    started: float = field(default_factory=time.time)
    ended: Optional[float] = None
    exit_code: Optional[int] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        d['cmd_str'] = ' '.join(self.cmd)
        d['elapsed'] = (self.ended or time.time()) - self.started
        return d


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.job: Optional[Job] = None
        self._proc: Optional[subprocess.Popen] = None
        self._lines: deque = deque(maxlen=MAX_LIVE_LINES)
        self._subs: List[queue.Queue] = []
        self._log_fh = None
        # parsed metrics for the current job
        self.metrics: Dict[str, Any] = self._empty_metrics()

    # ── metrics ────────────────────────────────────────────────────────────
    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            'iters': [], 'adv': [], 'strat': [], 'lr': [],
            'trav': [], 'train': [], 'buf': [],
            'iter': 0, 'total': 0, 'phase': '', 'filling': None,
            'selfplay': [],      # [{iter, af, fold, chk, call, bet}]
            'spots': [],         # [{iter, pct, exact, partial, n}]
            'checkpoints': [],
        }

    def _parse(self, line: str) -> None:
        m = _RE_ITER.search(line)
        if m:
            g = m.groupdict()
            it = int(g['iter'])
            mt = self.metrics
            mt['iter'], mt['total'] = it, int(g['total'])
            mt['iters'].append(it)
            mt['adv'].append(float(g['adv']))
            mt['strat'].append(float(g['strat']))
            mt['lr'].append(float(g['lr']))
            mt['trav'].append(float(g['trav']))
            mt['train'].append(float(g['train']))
            mt['buf'].append(int(g['buf']))
            mt['filling'] = None
        f = _RE_FILLING.search(line)
        if f:
            self.metrics['filling'] = {
                'have': int(f.group('have')), 'need': int(f.group('need'))}
        p = _RE_PHASE.search(line)
        if p:
            self.metrics['phase'] = f"P{p.group('n')}:{p.group('name')}"
        s = _RE_SELFPLAY.search(line)
        if s:
            g = s.groupdict()
            self.metrics['selfplay'].append({
                'iter': self.metrics['iter'], 'hands': int(g['hands']),
                'af': float(g['af']), 'fold': float(g['fold']),
                'chk': float(g['chk']), 'call': float(g['call']),
                'bet': float(g['bet']),
            })
        sp = _RE_SPOTS.search(line)
        if sp:
            g = sp.groupdict()
            self.metrics['spots'].append({
                'iter': self.metrics['iter'], 'n': int(g['n']),
                'pct': float(g['pct']), 'exact': int(g['exact']),
                'partial': int(g['partial']),
            })
        c = _RE_CKPT.search(line)
        if c and 'saved' in line.lower():
            self.metrics['checkpoints'].append(int(c.group(1)))

    # ── lifecycle ──────────────────────────────────────────────────────────
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, kind: str, label: str, cmd: List[str],
              params: Dict[str, Any], run_dir: Path) -> Job:
        with self._lock:
            if self.is_running():
                raise RuntimeError(
                    f'A job is already running ({self.job.kind}). '
                    f'Only one job may run at a time — they compete for GPU/CPU.')
            job = Job(id=run_dir.name, kind=kind, label=label,
                      cmd=cmd, params=params)
            self.job = job
            self._lines.clear()
            self.metrics = self._empty_metrics()
            self._log_fh = open(run_dir / 'output.log', 'w', buffering=1)

            env = dict(os.environ, PYTHONUNBUFFERED='1')
            self._proc = subprocess.Popen(
                cmd, cwd=str(REPO_ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                # own process group so we can signal the whole tree (workers!)
                start_new_session=True,
            )
            threading.Thread(target=self._pump, args=(job, run_dir),
                             daemon=True).start()
            self._emit(f'$ {" ".join(cmd)}', kind='meta')
            return job

    def _pump(self, job: Job, run_dir: Path) -> None:
        assert self._proc is not None
        stream = self._proc.stdout
        buf = ''
        try:
            while True:
                ch = stream.read(1)
                if not ch:
                    break
                # train_gpu.py uses \r for in-place progress; treat both as EOL
                if ch in '\r\n':
                    if buf.strip():
                        self._handle_line(buf)
                    buf = ''
                else:
                    buf += ch
            if buf.strip():
                self._handle_line(buf)
        except Exception as exc:                       # pragma: no cover
            self._emit(f'[runner] stream error: {exc}', kind='error')

        code = self._proc.wait()
        job.exit_code = code
        job.ended = time.time()
        if job.status == 'running':
            job.status = 'finished' if code == 0 else 'failed'
        self._emit(f'[runner] {job.status} (exit {code}) after '
                   f'{job.public()["elapsed"]:.0f}s', kind='meta')
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
        self._save_meta(job, run_dir)
        self._emit('__END__', kind='end')

    def _handle_line(self, line: str) -> None:
        self._parse(line)
        self._emit(line)

    def _emit(self, line: str, kind: str = 'log') -> None:
        rec = {'t': time.time(), 'line': line, 'kind': kind}
        if kind == 'log':
            self._lines.append(rec)
            if self._log_fh:
                try:
                    self._log_fh.write(line + '\n')
                except Exception:
                    pass
        for q in list(self._subs):
            try:
                q.put_nowait(rec)
            except queue.Full:
                pass

    def stop(self) -> bool:
        """SIGTERM the process group, escalate to SIGKILL if it lingers."""
        if not self.is_running():
            return False
        assert self._proc is not None
        if self.job:
            self.job.status = 'stopped'
        pgid = os.getpgid(self._proc.pid)
        os.killpg(pgid, signal.SIGTERM)
        for _ in range(50):                     # up to ~5s
            if self._proc.poll() is not None:
                return True
            time.sleep(0.1)
        os.killpg(pgid, signal.SIGKILL)
        return True

    # ── streaming ──────────────────────────────────────────────────────────
    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        if q in self._subs:
            self._subs.remove(q)

    def backlog(self) -> List[Dict[str, Any]]:
        return list(self._lines)

    # ── history ────────────────────────────────────────────────────────────
    @staticmethod
    def _save_meta(job: Job, run_dir: Path) -> None:
        try:
            (run_dir / 'job.json').write_text(json.dumps(job.public(), indent=2))
        except Exception:
            pass

    @staticmethod
    def new_run_dir(kind: str) -> Path:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        d = RUNS_DIR / f'{stamp}_{kind}_{uuid.uuid4().hex[:6]}'
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def history(limit: int = 40) -> List[Dict[str, Any]]:
        out = []
        for d in sorted(RUNS_DIR.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            meta = d / 'job.json'
            if meta.exists():
                try:
                    out.append(json.loads(meta.read_text()))
                except Exception:
                    continue
            if len(out) >= limit:
                break
        return out


MANAGER = JobManager()
