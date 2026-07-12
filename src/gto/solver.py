"""Drive the TexasSolver `console_solver` binary on batches of spots.

Interactive (stdin) mode doesn't process commands reliably, but the ~24s
dictionary load is paid ONCE per process invocation regardless of how many
solve blocks the input file contains. So the throughput strategy is: build
ONE command file with many solve blocks (each dumping to its own JSON), run
the binary once, read the dumps. A bad block (e.g. a range the binary rejects)
aborts the process mid-file, so `run_batch` re-runs the survivors adaptively:
the first missing dump after a run is the culprit — mark it errored and retry
the rest.

Binary location: `$TEXASSOLVER_BIN` or `tools/texassolver/console_solver`.
The binary needs its `resources/` dir, found relative to the binary.
"""
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from src.gto.action_mapping import bet_size_lines
from src.gto.spot_inputs import GTOSpotInputs

_DEFAULT_BIN = 'tools/texassolver/console_solver'


def _resolve_binary(binary: Optional[str]) -> str:
    return binary or os.environ.get('TEXASSOLVER_BIN') or _DEFAULT_BIN


@dataclass
class SolveSpec:
    """One solve to run: a unique id, the inputs, and both range strings."""
    solve_id: str
    inputs: GTOSpotInputs
    hero_range: str
    villain_range: str


@dataclass
class GTOSolver:
    binary: str = ''
    threads: int = 16
    accuracy: float = 0.5          # exploitability % target
    max_iteration: int = 80
    print_interval: int = 80
    dump_rounds: int = 1
    _resolved_bin: str = field(default='', init=False)

    def __post_init__(self):
        self._resolved_bin = _resolve_binary(self.binary or None)

    @property
    def available(self) -> bool:
        return Path(self._resolved_bin).exists()

    def version(self) -> str:
        return f"TexasSolver console_solver @ {self._resolved_bin}"

    def build_command_block(self, spec: SolveSpec, out_path: str) -> str:
        """One `set_*/build_tree/start_solve/dump_result` block for a spec.

        Hero's range is placed on the side hero plays (oop/ip); the villain
        range on the other side. Pot/effective-stack are the street-start
        values from `GTOSpotInputs` (chips).
        """
        inp = spec.inputs
        board_csv = ','.join(inp.board)
        if inp.hero_is_oop:
            oop_range, ip_range = spec.hero_range, spec.villain_range
        else:
            oop_range, ip_range = spec.villain_range, spec.hero_range
        lines = [
            f'set_pot {inp.pot_chips}',
            f'set_effective_stack {inp.eff_stack_chips}',
            f'set_board {board_csv}',
            f'set_range_ip {ip_range}',
            f'set_range_oop {oop_range}',
        ]
        # Inject the villain's exact bet size so the tree has a matching node.
        extra = {}
        if inp.villain_bet_side and inp.villain_bet_pct:
            extra[(inp.villain_bet_side, inp.street, 'bet')] = [inp.villain_bet_pct]
        lines.extend(bet_size_lines(inp.solve_streets, extra=extra))
        lines.extend([
            'build_tree',
            f'set_thread_num {self.threads}',
            f'set_accuracy {self.accuracy}',
            f'set_max_iteration {self.max_iteration}',
            f'set_print_interval {self.print_interval}',
            'set_use_isomorphism 1',
            'start_solve',
            f'set_dump_rounds {self.dump_rounds}',
            f'dump_result {out_path}',
        ])
        return '\n'.join(lines) + '\n'

    def run_batch(self, specs: List[SolveSpec],
                  timeout: int = 1800,
                  workdir: Optional[str] = None,
                  verbose: bool = False) -> Dict[str, dict]:
        """Solve all specs. Returns {solve_id: result_dict} for successes.

        Adaptive: if the binary aborts on a bad block, the first spec whose
        dump is missing is dropped (errored) and the rest retried, until all
        specs are resolved or a run makes no progress.
        """
        if not self.available:
            raise FileNotFoundError(
                f"console_solver not found at {self._resolved_bin}. "
                f"Run scripts/setup_texassolver.sh or set $TEXASSOLVER_BIN.")
        results: Dict[str, dict] = {}
        pending = list(specs)
        tmp = workdir or tempfile.mkdtemp(prefix='gto_solve_')
        os.makedirs(tmp, exist_ok=True)
        try:
            while pending:
                ran = self._run_once(pending, tmp, timeout, verbose)
                # Collect dumps in spec order; first missing = culprit.
                next_pending: List[SolveSpec] = []
                culprit_hit = False
                for spec in pending:
                    out_path = os.path.join(tmp, f'{spec.solve_id}.json')
                    res = _try_load(out_path)
                    if res is not None:
                        results[spec.solve_id] = res
                    elif not culprit_hit:
                        # First missing dump — it crashed the process. Drop it.
                        culprit_hit = True
                        if verbose:
                            print(f"  [solver] block {spec.solve_id} failed; "
                                  f"dropping and retrying {len(pending)-1} more")
                    else:
                        next_pending.append(spec)
                if not ran and not culprit_hit:
                    # Process didn't start / no dumps at all and nothing to drop.
                    break
                if len(next_pending) == len(pending):
                    break   # no progress — avoid infinite loop
                pending = next_pending
        finally:
            if workdir is None:
                shutil.rmtree(tmp, ignore_errors=True)
        return results

    def _run_once(self, specs: List[SolveSpec], tmp: str,
                  timeout: int, verbose: bool) -> bool:
        """Write a multi-block command file and run the binary once."""
        for spec in specs:
            out_path = os.path.join(tmp, f'{spec.solve_id}.json')
            if os.path.exists(out_path):
                os.remove(out_path)
        cmd_file = os.path.join(tmp, 'batch_commands.txt')
        with open(cmd_file, 'w') as f:
            for spec in specs:
                out_path = os.path.join(tmp, f'{spec.solve_id}.json')
                f.write(self.build_command_block(spec, out_path))
        bin_abs = Path(self._resolved_bin).resolve()
        resource_dir = str(bin_abs.parent / 'resources')
        argv = [str(bin_abs), '--input_file', cmd_file,
                '--resource_dir', resource_dir, '--mode', 'holdem']
        try:
            proc = subprocess.run(
                argv, stdout=subprocess.DEVNULL,
                stderr=(None if verbose else subprocess.DEVNULL),
                timeout=timeout,
                cwd=str(bin_abs.parent))
            return proc.returncode == 0 or any(
                os.path.exists(os.path.join(tmp, f'{s.solve_id}.json'))
                for s in specs)
        except subprocess.TimeoutExpired:
            if verbose:
                print(f"  [solver] batch timed out after {timeout}s")
            return any(os.path.exists(os.path.join(tmp, f'{s.solve_id}.json'))
                       for s in specs)


def _try_load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None
