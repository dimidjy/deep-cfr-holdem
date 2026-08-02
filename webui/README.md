# Poker Runner — web UI

A small job runner for this project's scripts. Fill a form, hit run, watch the
output and the metrics live in a browser.

## Start

```bash
python3 -m webui.server                      # binds 0.0.0.0:8000
python3 -m webui.server --port 9000
python3 -m webui.server --host 127.0.0.1     # local only (recommended)
```

**There is no authentication.** Anyone who can reach the port can start a
multi-day GPU job. Bind it to localhost and use an SSH tunnel:

```bash
ssh -L 8000:localhost:8000 user@server
# then open http://localhost:8000
```

To keep it up across disconnects, run it under `tmux`, or as a systemd unit.

## What it does

**Training tab** — every knob `scripts/train_gpu.py` accepts, pre-filled from
`configs/training_config.json` so the form always matches what the script would
do on its own. Optimizer settings, imitation-learning strength, all reward
shapers, opponent settings, and the K-Best pool.

**Validation tab** — self-play GTO check, hand-history validation, spot
inspection, and annotated-spot scoring. `--betting-line` is added automatically
for hand-history runs, since 53-dim models crash without it.

**Live panels** — iteration/ETA progress, advantage and strategy loss chart,
self-play action mix coloured against the GTO bands (AF 2.0–3.5, bet 55–70%),
annotated spot score, and the raw console.

## One job at a time

The runner refuses to start a second job while one is running (HTTP 409).
That is deliberate: training and validation scripts compete for the same GPU
and CPU cores, and running them together corrupts timings and starves the
trainer. `scripts/tournament.py` is the documented exception — it parallelises
internally via `--workers`, so it still counts as one job here.

Stop sends `SIGTERM` to the whole process group (so worker processes die too),
then escalates to `SIGKILL` after 5s.

## Run artifacts

Each run gets `webui/runs/<timestamp>_<kind>_<id>/` containing:

- `config.json` — the exact generated config that was passed to the script
- `output.log` — full stdout/stderr
- `job.json` — command line, parameters, status, exit code, duration

So any run can be audited or reproduced later from its own directory.

## Notes surfaced in the UI

- Pass an equity model. Without it the engine computes `obs[18]` as heads-up
  equity even in multiway pots, which is not what the runtime observation builder computes.
- Stop training if self-play bet% drifts above 70% — that is the known
  over-aggression failure mode; pick the last in-band checkpoint.
- Keep each reward shaper ≤ 0.5.

## Adding another script

Add a branch in `run_validate()` in `webui/server.py` building the argv, and an
option in the `v_kind` select in `webui/static/index.html`. If the script prints
a metric you want charted, add a regex to `webui/jobs.py` next to the existing
`_RE_ITER` / `_RE_SELFPLAY` / `_RE_SPOTS` patterns.
