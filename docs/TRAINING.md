# Training

Run one training or validation job at a time; they compete for GPU and CPU.
`scripts/tournament.py` parallelizes its own subprocesses (`--workers N`).

## Deep CFR

```bash
python3 scripts/train_gpu.py --config configs/training_config.json \
    --equity-model models/equity_model_v3.pt
```

- External-sampling MCCFR traversals in worker processes, with exact regrets at
  every traversal-player decision. Turn/river branches use
  5 rollouts and flop/preflop use 3.
- Residual advantage and strategy networks. The strategy net returns logits
  and masked softmax happens in `DeepCFR`.
- Reservoir-sampled replay buffers keep a uniform sample over all
  iterations instead of a sliding window.
- Strategy-net targets are ε-free. Exploration is applied only when
  sampling.
- Always pass `--equity-model`. Without it the engine falls back to Monte
  Carlo equity, which is slower and noisier.

### Reward shaping

All shapers are config/CLI flags, default off unless validated:

| Flag | Effect |
|---|---|
| `--bluff-penalty` | penalize raising weak hands |
| `--call-bonus` | reward +EV calls (equity > pot odds) |
| `--value-bonus` | reward betting strong hands when checked to (river requires a made pair) |
| `--draw-call-boost`, `--draw-bet-boost` | draw-aware bonuses, auto-disabled on the river |
| `--deck-reshuffle` | variance reduction: re-randomize future cards per branch |

### Imitation on annotated spots

`--imitation-weight` adds a cross-entropy term toward labeled spot actions
after each CFR pass. The current setup is weight 0.3, 5 passes, batch 256,
with river ×3 / turn ×1.5 / fold-label ×2 oversampling. Labels come from
`scripts/annotate_spots_with_claude.py` (LLM-assisted, then checked with
`verify_claude_annotations.py`). Refresh stored observations after
changing a builder:
`python3 scripts/rebuild_spot_obs.py --annotated-only --force --execute`.

## Burst-loop training with a K-best pool

```bash
python3 scripts/train_burst_loop.py --start-checkpoint <ckpt> --start-iter <n> \
    --kbest-seed <m1.pt> <m2.pt> ... --workers 32
```

Each cycle trains a short burst, evaluates the new checkpoints and keeps the
best ones in the opponent pool. Seed the pool across the aggression axis
(passive → balanced → aggressive) so the trainee learns to defend against
every style.

## Offline RL from hand histories

```bash
python3 scripts/build_training_data.py ...   # decision points + spot DB
python3 scripts/train_supervised.py --hh-dir <dir> --good-players <file>
python3 scripts/train_cql.py --config configs/cql_config.json
python3 scripts/train_iql.py --config configs/cql_config.json
```

The cohort comes from `scripts/compute_player_winrates.py` (minimum hand
count and win-rate thresholds). Supervised training supports fold weighting,
counterfactual folds and reward weighting.

## Auxiliary models

- **Equity network** (`scripts/train_equity_model.py`): predicts multiway
  equity from hand and board encodings and replaces Monte Carlo equity in
  the hot path. Inference is batched and protected by a lock.
- **Opponent range estimator** (`scripts/train_range_estimator.py`).
- **Re-raise classifier** (`scripts/train_reraise.py`).
- **Model soup** (`scripts/make_model_soup.py`): weight averaging across
  sibling checkpoints.

## Model selection

1. Self-play statistics must sit inside the postflop target bands:
   CBet 55–70%, AF 2.0–3.5, WTSD 25–32%.
2. Within those bands, maximize head-to-head edge in the tournament.
3. Use hand-history agreement as a floor check (doesn't fold the nuts,
   doesn't call down with air), not as an optimization target.

Training drifts toward over-aggression after roughly 3,400 iterations.
When CBet goes above 70%, stop and take the last in-band checkpoint.
